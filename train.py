import os
import re
import csv
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import nibabel as nib  # for reading NIfTI files

# ================================
# 1. CSV Reading Functions
# ================================
def load_scan_summary(csv_path):
    """
    Reads the two_valid_scans_summary.csv file.
    Expected columns: Subject, First_Scan_Date, T2_First, Second_Scan_Date, T2_Second.
    Returns a dict mapping subject -> [first_scan, second_scan].
    """
    scan_summary = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subject = row["Subject"].strip()
            first_scan = row["First_Scan_Date"].strip()
            second_scan = row["Second_Scan_Date"].strip()
            scan_summary[subject] = [first_scan, second_scan]
    return scan_summary

def load_subject_dictionary(csv_path):
    """
    Reads subject_dictionary.csv.
    Expected columns include: SUBJECTID and PCR.
    Uses the PCR column as the binary label.
    Returns a dict mapping subject -> label.
    """
    subject_dict = {}
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subject = row["SUBJECTID"].strip()
            try:
                pcr_value = int(row["PCR"].strip())
            except:
                pcr_value = 0
            subject_dict[subject] = pcr_value
    return subject_dict

# ================================
# 2. Helper Functions and Synonym Lists
# ================================
def parse_date_from_folder(folder_name):
    """
    Parses a date from the first 10 characters of a folder name (MM-DD-YYYY).
    If parsing fails, returns datetime.max.
    """
    try:
        date_str = folder_name[:10]
        return datetime.strptime(date_str, "%m-%d-%Y")
    except Exception:
        return datetime.max

# Synonym lists for modality detection:
SEQ_SYNONYMS = {
    "SER": ["ser"],
    "T2": ["t2", "tset2", "fse t2", "t2-fse"],
    "DCE": [
        "dynamic-3dfgre", "ir-spgr", "spgr", "vibe", "fl3dsag", "breas3dsag",
        "t1sag", "dynamic", "dce", "3d-", "t13dse", "delaycontra", "post",
        "delay", "highressag", "fatsat", "fl3d", "unisag", "post-",
        "left breast post", "6dyn", "bind", "3dsag6d"
    ]
}

def extract_first_digit(name):
    """Returns the first digit found in name (as string) or None."""
    if name is None:
        return None
    for ch in name:
        if ch.isdigit():
            return ch
    return None

# ================================
# 3. DCE 3D → 2D Conversion with Split and MIP
# ================================
def load_dce_as_mip(subject, scan_date, dce_folder, dce_filename, ser_file_count, target_size=(299,299), base_dir="/scratch/l.peiwang/ISPY_processed"):
    """
    Loads the 3D DCE NIfTI file from:
       /scratch/l.peiwang/ISPY_processed/<subject>/<scan_date>/<dce_folder>/<dce_filename>
    Assumes the DCE volume's slice dimension is approximately ser_file_count * 3.
    Splits the volume into three groups (each group size = ser_file_count) and computes a
    Maximum Intensity Projection (MIP) for each group.
    Each resulting 2D image is resized to target_size and its single channel is replicated to 3 channels.
    If the total slice count deviates significantly from ser_file_count*3, a warning is printed.
    Returns a list of 3 2D images.
    """
    file_path = os.path.join(base_dir, subject, scan_date, dce_folder, dce_filename)
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
    except Exception as e:
        print(f"Error loading DCE file for {subject}, {scan_date}, {dce_folder}: {e}")
        return [np.zeros((*target_size, 3), dtype=np.float32)] * 3
    if data.ndim == 4:
        data = data[:,:,:,0]
    total_slices = data.shape[2]
    expected = ser_file_count * 3
    if abs(total_slices - expected) > 1:
        print(f"Warning: For {subject} {scan_date}, DCE slices = {total_slices} but expected ≈ {expected} (SER count = {ser_file_count}).")
    group_size = ser_file_count  # each group should have ser_file_count slices
    imgs = []
    for i in range(3):
        start = i * group_size
        end = start + group_size
        if end > total_slices:
            end = total_slices
        group = data[:, :, start:end]
        # Compute MIP along the slice axis
        mip = np.max(group, axis=2)
        mip_tensor = tf.convert_to_tensor(mip[..., np.newaxis], dtype=tf.float32)
        mip_resized = tf.image.resize(mip_tensor, target_size)
        mip_resized = tf.repeat(mip_resized, repeats=3, axis=-1)
        imgs.append(mip_resized.numpy())
    return imgs

# ================================
# 4. Build Modalities for a Given Subject & Scan Date
# ================================
def build_modalities_for_subject(subject, scan_date, base_dir="/scratch/l.peiwang/ISPY_processed"):
    """
    For a given subject and scan_date, directly searches the files in the directory
    for NIfTI files whose names contain modality synonyms.
    
    For SER: returns the first file name that contains any SER synonym.
    For T2: returns the first file name that contains any T2 synonym.
    For DCE: returns the file name that contains any DCE synonym, provided its first digit
             matches that of the SER file.
    Also returns the SER file count based on the number of .dcm files in the directory.
    """
    subject_scan_dir = os.path.join(base_dir, subject, scan_date)
    files = [f for f in os.listdir(subject_scan_dir) if f.lower().endswith(".nii.gz")]
    ser_file = None
    t2_file = None
    dce_file = None
    ser_count = 0
    for file in sorted(files):
        lower = file.lower()
        if ser_file is None and any(kw in lower for kw in SEQ_SYNONYMS["SER"]):
            ser_file = file
            dcm_files = [x for x in os.listdir(subject_scan_dir) if x.lower().endswith(".dcm") and any(kw in x.lower() for kw in SEQ_SYNONYMS["SER"])]
            ser_count = len(dcm_files) if dcm_files else 40
        if t2_file is None and any(kw in lower for kw in SEQ_SYNONYMS["T2"]):
            t2_file = file
        if dce_file is None and any(kw in lower for kw in SEQ_SYNONYMS["DCE"]):
            dce_file = file
    ser_digit = extract_first_digit(ser_file)
    if dce_file and ser_digit:
        if extract_first_digit(dce_file) != ser_digit:
            print(f"Warning: For {subject} {scan_date}, SER file '{ser_file}' and DCE file '{dce_file}' mismatch in first digit.")
            dce_file = None
    return {
        "SER": ser_file,
        "T2": t2_file,
        "DCE_filename": dce_file,
        "SER_count": ser_count
    }

# ================================
# 5. Data Generator from Modalities Dictionary
# ================================
def data_generator_from_modalities(subject_list, scan_date_dict, subject_dict, modality_keys, avg_images, batch_size=4, base_dir="/scratch/l.peiwang/ISPY_processed", selected_scan_indices=[0,1]):
    """
    Generator that yields batches of inputs and labels.
    For each subject (from scan_date_dict) and for each scan date specified by selected_scan_indices,
    it:
      - Loads SER: directly searches for a NIfTI file containing a SER synonym.
      - Loads T2 similarly.
      - Loads DCE: directly searches for a NIfTI file containing a DCE synonym (with matching first digit to SER),
        then calls load_dce_as_mip to split the 3D volume into 3 2D images.
    The final input order is constructed based on the selected modalities for each scan date.
    For each scan date, if a modality is in modality_keys:
         - "SER" → 1 image.
         - "DCE" → 3 images.
         - "T2"  → 1 image.
    With two scan dates (default), the outputs are concatenated accordingly.
    Missing modalities are replaced by the corresponding average images.
    """
    # Build the output order for one scan date based on modality_keys:
    single_scan_order = []
    for key in modality_keys:
        if key.upper() == "DCE":
            single_scan_order.extend(["DCE1", "DCE2", "DCE3"])
        else:
            single_scan_order.append(key.upper())
    # For each selected scan date, append a suffix:
    total_order = [f"{mod}_{i+1}" for i in selected_scan_indices for mod in single_scan_order]
    while True:
        inputs_batch = [[] for _ in range(len(total_order))]
        labels = []
        for _ in range(batch_size):
            subject = np.random.choice(subject_list)
            label = subject_dict[subject]
            dates = scan_date_dict[subject]
            imgs = []
            for idx in selected_scan_indices:
                if idx >= len(dates):
                    continue
                scan_date = dates[idx]
                mod_info = build_modalities_for_subject(subject, scan_date, base_dir)
                scan_imgs = []
                # SER:
                if "SER" in modality_keys:
                    if mod_info["SER"]:
                        ser_path = os.path.join(base_dir, subject, scan_date, mod_info["SER"])
                        try:
                            ser_img = tf.keras.preprocessing.image.load_img(ser_path, target_size=(299,299))
                            ser_img = tf.keras.preprocessing.image.img_to_array(ser_img)/255.0
                        except Exception as e:
                            print(f"Error loading SER for {subject} {scan_date}: {e}")
                            ser_img = avg_images.get("SER", np.zeros((299,299,3), dtype=np.float32))
                    else:
                        ser_img = avg_images.get("SER", np.zeros((299,299,3), dtype=np.float32))
                    scan_imgs.append(ser_img)
                # DCE:
                if "DCE" in modality_keys:
                    if mod_info["DCE_filename"]:
                        dce_imgs = load_dce_as_mip(subject, scan_date, "", mod_info["DCE_filename"], mod_info["SER_count"], target_size=(299,299), base_dir=base_dir)
                        if dce_imgs is None or len(dce_imgs) != 3:
                            print(f"Error: For {subject} {scan_date}, DCE splitting failed. Using average DCE image.")
                            dce_imgs = [avg_images.get("DCE", np.zeros((299,299,3), dtype=np.float32))]*3
                    else:
                        dce_imgs = [avg_images.get("DCE", np.zeros((299,299,3), dtype=np.float32))]*3
                    scan_imgs.extend(dce_imgs)
                # T2:
                if "T2" in modality_keys:
                    if mod_info["T2"]:
                        t2_path = os.path.join(base_dir, subject, scan_date, mod_info["T2"])
                        try:
                            t2_img = tf.keras.preprocessing.image.load_img(t2_path, target_size=(299,299))
                            t2_img = tf.keras.preprocessing.image.img_to_array(t2_img)/255.0
                        except Exception as e:
                            print(f"Error loading T2 for {subject} {scan_date}: {e}")
                            t2_img = avg_images.get("T2", np.zeros((299,299,3), dtype=np.float32))
                    else:
                        t2_img = avg_images.get("T2", np.zeros((299,299,3), dtype=np.float32))
                    scan_imgs.append(t2_img)
                imgs.extend(scan_imgs)
            for i in range(len(imgs)):
                inputs_batch[i].append(imgs[i])
            labels.append(label)
        inputs_batch = [np.array(x) for x in inputs_batch]
        labels = np.array(labels)
        yield inputs_batch, labels

# ================================
# 6. Compute Average Images for Modality Keys
# ================================
def compute_average_images_from_modalities(subject_list, scan_date_dict, modality_keys, base_dir="/scratch/l.peiwang/ISPY_processed"):
    """
    Computes the average image for each modality key ("SER", "T2", "DCE") over the given subjects.
    For DCE, uses load_dce_as_mip to obtain a representative 2D image.
    Returns a dict mapping modality key -> average image (shape (299,299,3)).
    """
    avg_images = {key: np.zeros((299,299,3), dtype=np.float32) for key in modality_keys}
    counts = {key: 0 for key in modality_keys}
    for subject in subject_list:
        subject_dir = os.path.join(base_dir, subject)
        dates = scan_date_dict.get(subject, [])
        if len(dates) < 1:
            continue
        for scan in dates:
            scan_dir = os.path.join(subject_dir, scan)
            if not os.path.isdir(scan_dir):
                continue
            mod_info = build_modalities_for_subject(subject, scan, base_dir)
            # SER:
            if "SER" in modality_keys and mod_info["SER"]:
                ser_path = os.path.join(scan_dir, mod_info["SER"])
                try:
                    img = tf.keras.preprocessing.image.load_img(ser_path, target_size=(299,299))
                    img = tf.keras.preprocessing.image.img_to_array(img)/255.0
                    avg_images["SER"] += img
                    counts["SER"] += 1
                except Exception as e:
                    print(f"Error loading SER for average: {subject}, {scan}: {e}")
            # T2:
            if "T2" in modality_keys and mod_info["T2"]:
                t2_path = os.path.join(scan_dir, mod_info["T2"])
                try:
                    img = tf.keras.preprocessing.image.load_img(t2_path, target_size=(299,299))
                    img = tf.keras.preprocessing.image.img_to_array(img)/255.0
                    avg_images["T2"] += img
                    counts["T2"] += 1
                except Exception as e:
                    print(f"Error loading T2 for average: {subject}, {scan}: {e}")
            # DCE:
            if "DCE" in modality_keys and mod_info["DCE_filename"]:
                dce_img = load_dce_as_mip(subject, scan, "", mod_info["DCE_filename"], mod_info["SER_count"], target_size=(299,299), base_dir=base_dir)
                avg_images["DCE"] += dce_img
                counts["DCE"] += 1
    for key in modality_keys:
        if counts[key] > 0:
            avg_images[key] /= counts[key]
        else:
            avg_images[key] = np.zeros((299,299,3), dtype=np.float32)
    return avg_images

# ================================
# 7. Build Multimodal Model
# ================================
def build_multimodal_model(modalities_total, input_shape):
    """
    Builds a multimodal model with one branch per input.
    modalities_total is a list of names for the inputs (e.g.,
    ["SER_1", "DCE1_1", "DCE2_1", "DCE3_1", "T2_1", "SER_2", "DCE1_2", "DCE2_2", "DCE3_2", "T2_2"]).
    """
    branches = []
    inputs = []
    for mod in modalities_total:
        inp = Input(shape=input_shape, name=f"{mod}_input")
        inputs.append(inp)
        base_model = InceptionV3(weights="imagenet", include_top=False, input_tensor=inp)
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        branches.append(x)
    combined = Concatenate()(branches)
    x = Dropout(0.5)(combined)
    output = Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ================================
# 8. Evaluation Function
# ================================
def evaluate_model(model, generator, steps):
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    y_true = []
    y_pred = []
    for _ in range(steps):
        x_batch, labels = next(generator)
        preds = model.predict(x_batch)
        y_true.extend(labels)
        y_pred.extend(preds.flatten())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_bin = (y_pred >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_bin)
    auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred_bin)
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
        PPV = TP/(TP+FP) if (TP+FP) > 0 else 0
        NPV = TN/(TN+FN) if (TN+FN) > 0 else 0
    else:
        PPV = NPV = 0
    return acc, auc, PPV, NPV, cm

# ================================
# 9. Main Pipeline
# ================================
def main():
    # 1. Read the two CSV files.
    scan_summary = load_scan_summary("two_valid_scans_summary.csv")
    subject_dict = load_subject_dictionary("Dictionary.csv")
    
    # 2. Use scan_summary as our scan_date_dict.
    scan_date_dict = scan_summary
    subject_list = list(scan_date_dict.keys())
    print(f"Total subjects from summary: {len(subject_list)}")
    
    # 3. Split subjects into train (60%), val (20%), test (20%).
    np.random.shuffle(subject_list)
    n = len(subject_list)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_subjects = subject_list[:train_end]
    val_subjects = subject_list[train_end:val_end]
    test_subjects = subject_list[val_end:]
    print(f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")
    
    # 4. Define modality keys and scan date selection.
    # For each scan date, if selected_modality_keys is ["SER", "DCE", "T2"], then per scan date we get:
    # 1 SER, 3 DCE splits, 1 T2 = 5 outputs.
    # We now also allow selection of scan dates via selected_scan_indices.
    selected_modality_keys = ["SER"]
    #selected_modality_keys = ["SER", "DCE", "T2"]
    # Set which scan dates to use; default is [0,1] (both scan dates)
    #selected_scan_indices = [0, 1]
    selected_scan_indices = [0]
    single_scan_order = []
    for key in selected_modality_keys:
        if key.upper() == "DCE":
            single_scan_order.extend(["DCE1", "DCE2", "DCE3"])
        else:
            single_scan_order.append(key.upper())
    modalities_total = [f"{mod}_{i+1}" for i in selected_scan_indices for mod in single_scan_order]
    
    # 5. Compute average images for each modality key from training subjects.
    avg_images = compute_average_images_from_modalities(train_subjects, scan_date_dict, selected_modality_keys)
    
    # 6. Build the multimodal model.
    input_shape = (299, 299, 3)
    model = build_multimodal_model(modalities_total, input_shape)
    model.summary()
    
    # 7. Create data generators.
    batch_size = 4
    train_gen = data_generator_from_modalities(train_subjects, scan_date_dict, subject_dict, selected_modality_keys, avg_images, batch_size=batch_size, selected_scan_indices=selected_scan_indices)
    val_gen = data_generator_from_modalities(val_subjects, scan_date_dict, subject_dict, selected_modality_keys, avg_images, batch_size=batch_size, selected_scan_indices=selected_scan_indices)
    test_gen = data_generator_from_modalities(test_subjects, scan_date_dict, subject_dict, selected_modality_keys, avg_images, batch_size=batch_size, selected_scan_indices=selected_scan_indices)
    
    # 8. Train the model.
    epochs = 10
    steps_per_epoch = 20
    validation_steps = 10
    callbacks = [
        ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    ]
    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch,
              validation_data=val_gen, validation_steps=validation_steps,
              callbacks=callbacks)
    
    # 9. Evaluate performance on the test set.
    test_steps = 20
    acc, auc, PPV, NPV, cm = evaluate_model(model, test_gen, steps=test_steps)
    print("\n===== PERFORMANCE EVALUATION ON TEST SET =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"PPV (Precision): {PPV:.4f}")
    print(f"NPV: {NPV:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()

