import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger

from rnn_examples.names_dataset import NamesDataset


class RnnModel(pl.LightningModule):
    def __init__(self, vocab_size, n_classes, dset):
        super(RnnModel, self).__init__()

        self.embed = nn.Embedding.from_pretrained(embeddings=torch.eye(vocab_size))
        self.rnn = nn.GRU(input_size=vocab_size, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, n_classes)
        self.dset = dset

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embed(x)
        x, h = self.rnn(x)
        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)

        y_hat = output[:, -1, :]

        loss = self.criterion(y_hat, y)

        return {"loss": loss, "log": {"train_loss": loss}}

    def configure_optimizers(self):
        return Adam(self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dset, batch_size=1, shuffle=True)


logger = TensorBoardLogger("runs")

dset = NamesDataset(Path("data/names"))
model = RnnModel(dset.get_vocab_size(), dset.get_n_classes(), dset)
trainer = pl.Trainer(
    min_epochs=5,
    max_epochs=10,
    early_stop_callback=None,
    gradient_clip=1.0,
    accumulate_grad_batches=16,
    gpus=[0],
)
trainer.fit(model)
