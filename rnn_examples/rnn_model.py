import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger

from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.utils.rnn as ru
from sklearn.preprocessing import LabelBinarizer, label_binarize
from rnn_examples.names_dataset import NamesDataset


class RnnModel(pl.LightningModule):
    def __init__(self, vocab_size, n_classes, dset):
        super(RnnModel, self).__init__()

        embed_tensor = torch.eye(vocab_size)
        embed_tensor[0, :] = 0
        self.embed = nn.Embedding.from_pretrained(embeddings=torch.eye(vocab_size))
        self.rnn = nn.GRU(input_size=vocab_size, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, n_classes)
        self.dset = dset
        self.n_classes = n_classes

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, l):

        padded = ru.pad_sequence(x, batch_first=True)

        embedded = self.embed(padded)

        packed = ru.pack_padded_sequence(embedded, l,batch_first=True, enforce_sorted=False)

        x, h = self.rnn(packed)

        unpacked, sizes = ru.pad_packed_sequence(x, batch_first=True)

        sizes = sizes-1
        i = sizes.repeat(unpacked.size(-1)).view(-1,unpacked.size(0)).transpose(0,1).unsqueeze(1)
        s = torch.gather(unpacked, 1, i)

        s = s.squeeze()


        # x = unpacked[:,-1,:].squeeze()

        x = self.fc(s)

        return x

    def training_step(self, batch, batch_idx):
        x, y, l = batch

        output = self.forward(x, l)

        y_hat = output

        loss = self.criterion(y_hat, y)

        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        x, y, l = batch

        output = self.forward(x, l)

        y_hat = output

        loss = self.criterion(y_hat, y)
 
        # get predictions
        preds = torch.argmax(output, dim=1)
        
        # get correct predictions
        correct = (preds == y).float()
        
        log = {
            'val_loss': loss,
            'correct': correct,
            'logits': output,
            'trues': y,
            'preds': preds
        }

        return log
        
    def validation_end(self, outputs):
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.cat([x['correct'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs]).cpu()
        trues = torch.cat([x['trues'] for x in outputs]).cpu().numpy()
        preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        
        binarized_trues = label_binarize(trues, list(range(self.n_classes)))

        scores = torch.softmax(logits, dim=1).numpy()
        micro_f1 = f1_score(trues, preds, average='micro')
        micro_auc = roc_auc_score(binarized_trues, scores, average='micro')
        macro_auc = roc_auc_score(binarized_trues, scores, average='macro')
        
        tensorboard_logs = {'val_loss': avg_loss,
                           'val_acc': val_acc,
                           'micro_f1': torch.FloatTensor([micro_f1.item()]),
                           'micro_auc': micro_auc,
                           'macro_auc': macro_auc,
                           }
        print(tensorboard_logs)
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        return Adam(self.parameters())

    def collate(self, batch):
        x, y, l = zip(*batch)

        x = list(x)
        y = torch.LongTensor(y)
        l = torch.LongTensor(l)

        return x, y, l

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dset, batch_size=16, shuffle=True, collate_fn=self.collate)

    @pl.data_loader
    def val_dataloader(self):
        loader = DataLoader(self.dset, batch_size=16, shuffle=False, collate_fn=self.collate)
        return loader


logger = TensorBoardLogger("namesruns")

dset = NamesDataset(Path("data/names"))
model = RnnModel(dset.get_vocab_size(), dset.get_n_classes(), dset)
trainer = pl.Trainer(
    logger=logger,
    min_epochs=5,
    max_epochs=10,
    early_stop_callback=None,
    gradient_clip=1.0,
    num_sanity_val_steps=0,
    # val_check_interval=0.2,
    # accumulate_grad_batches=16,
    gpus=None,
)
trainer.fit(model)
