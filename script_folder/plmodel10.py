import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import pytorch_lightning as pl


class LitTrainClassifier(pl.LightningModule):

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

class LitValidationClassifier(pl.LightningModule):

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_pred = torch.argmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

class LitTestClassifier(pl.LightningModule):

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_pred = torch.argmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_pred, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

class LSTMModel(LitTrainClassifier, LitValidationClassifier, LitTestClassifier):

    def __init__(self, vocab_size=7295, embedding_dim=200, hidden_dim=100, layer_dim=2, output_dim=9, drop_out=0.3):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        self.lstm = torch.nn.LSTM(input_size = embedding_dim,
                                  hidden_size = hidden_dim,
                                  num_layers = layer_dim,
                                  dropout = drop_out,
                                  batch_first=True)    
            
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
    def forward(self, x):
        x = self.embeddings(x)        
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
