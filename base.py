import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score


class Base(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.val_acc = []

    def validation_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1)
        loss = F.cross_entropy(output, target) * data.size(0)

        return pred, target, loss

    def validation_epoch_end(self, outputs):
        pred, truth = [], []
        n_samples = 0
        loss = 0
        for p, t, l in outputs:
            pred.extend(p.tolist())
            truth.extend(t.tolist())
            loss += l
            n_samples += p.size(0)
        accuracy, _, _ = self.balanced_acc(truth, pred)
        loss = loss.item()/n_samples
        self.log("val_acc", accuracy)
        self.log("val_loss", loss)
        self.val_acc.append(accuracy)

    def test_step(self, batch, batch_nb):
        data, target = batch
        return self.forward(data), target

    def test_epoch_end(self, outputs):
        pred, score, truth = [], [], []
        for out, y in outputs:
            p = out.argmax(dim=-1)
            pred.extend(p.cpu().tolist())

            p = out.softmax(dim=-1)               
            score.extend(p[:, 1].cpu().tolist())

            truth.extend(y.tolist())

        accuracy, sensitivity, specificity = self.balanced_acc(truth, pred)
        auc = roc_auc_score(truth, score)

        self.test_results = {
            "accuracy": accuracy,
            "val_accuracy": max(self.val_acc),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc
        }

    def balanced_acc(self, truth, pred):
        report = classification_report(truth, pred, output_dict=True)
        bacc = report["macro avg"]["recall"]
        # Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.
        sensitivity = report["1"]["recall"]  # AD
        specificity = report["0"]["recall"]  # CN
        return bacc, sensitivity, specificity
