import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from flair.embeddings import WordEmbeddings
from flair.datasets import IMDB

import pytorch_lightning as pl

from pytorch_lightning.logging  import TensorBoardLogger

class GloveIMDBDataset(Dataset):
    def __init__(self, is_train):
        self.glove_embedding = WordEmbeddings('glove')
        self.dset = IMDB().train if is_train else IMDB().test
        # glove_embedding.embed(self.dset)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        embedded = self.glove_embedding.embed(self.dset[idx])[0]

        tokens = torch.stack([t.embedding for t in embedded])

        y = 1.0 if embedded.labels[0].value =='pos' else 0.0

        return tokens, y

class RnnModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.rnn = nn.GRU(input_size=100, hidden_size=64, batch_first=True)

        self.fc = nn.Linear(64, 1)

        self.criterion = nn.BCELoss()

    def forward(self, x):
        packed = pack_sequence(x, enforce_sorted=False)

        rnn_out, rnn_hidden = self.rnn(packed)

        unpacked, sizes = pad_packed_sequence(rnn_out, batch_first=True)

        sizes = sizes-1
        i = sizes.repeat(unpacked.size(-1)).view(-1,unpacked.size(0)).transpose(0,1).unsqueeze(1)
        s = torch.gather(unpacked, 1, i)

        s = s.squeeze()

        y = self.fc(s)

        return F.sigmoid(y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)

        output = output.squeeze()

        y_hat = output

        loss = self.criterion(y_hat, y)

        return {"loss": loss, "log": {"train_loss": loss}}
    def configure_optimizers(self):
        return Adam(self.parameters())

    def collate(self, batch):
        x, y = zip(*batch)

        x = list(x)
        y = torch.FloatTensor(y)

        return x, y


    @pl.data_loader
    def train_dataloader(self):
        train_dset = GloveIMDBDataset(is_train=True)
        return DataLoader(train_dset, batch_size=16, shuffle=True, collate_fn=self.collate)

    


logger = TensorBoardLogger("imdb")

model = RnnModel()
trainer = pl.Trainer(
    logger=logger,
    min_epochs=1,
    max_epochs=1,
    early_stop_callback=None,
    gradient_clip=1.0,
    num_sanity_val_steps=5,
    # val_check_interval=0.2,
    # accumulate_grad_batches=16,
    gpus=None,
    resume_from_checkpoint=None
)
trainer.fit(model)



