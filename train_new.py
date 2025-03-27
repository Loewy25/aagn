import argparse
import random

import torch
from torchvision.transforms import Compose
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataset import ADNIDataset
from transform import FlipBrain, MinMaxInstance, GaussianBlur
from aagn import AAGN


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="aagn", help="id for current run")
    parser.add_argument("--finetune", action="store_true", help="Flag to load a checkpoint and fine tune")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoint file")
    args = parser.parse_args()
    
    train_transforms = Compose([MinMaxInstance(), FlipBrain(), GaussianBlur()])
    test_transforms = Compose([MinMaxInstance()])    
    
    pl.seed_everything(42)
    
    if args.finetune and args.ckpt_path:
        model = AAGN.load_from_checkpoint(args.ckpt_path)
        print(f"Loaded model from {args.ckpt_path} for fine tuning.")
    else:
        model = AAGN()
    
    hparams = model.hparams

    train_dataset = ADNIDataset("data/train (3).csv", train_transforms)    
    val_dataset = ADNIDataset("data/val (1).csv", test_transforms)
    test_dataset = ADNIDataset("data/test (7).csv", test_transforms)
        
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=16,
        worker_init_fn=seed_worker,
        drop_last=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=4,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    checkpoint_callback = ModelCheckpoint(
        filename=f"{args.run_id}", 
        verbose=True,
        monitor="val_acc",
        mode="max",
    )

    trainer = pl.Trainer(
        default_root_dir="logs",
        logger=pl.loggers.TensorBoardLogger("logs/", name=args.run_id),
        max_epochs=hparams.epoch,
        accelerator="gpu",
        devices=[0],
        num_sanity_val_steps=0,
        deterministic=True,
        precision=16,
        callbacks=[checkpoint_callback],
    )


    if args.finetune and args.ckpt_path:
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, train_dataloader, val_dataloader)
    
    trainer.test(ckpt_path='best', dataloaders=test_dataloader)[0]
    
    print("================== RESULTS ==================")
    print(trainer.model.test_results)
