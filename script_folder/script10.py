import argparse
import os
import numpy as np
from azureml.core import Run
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from plmodel10 import LSTMModel

run = Run.get_context()
ws = run.experiment.workspace
mlflow_url = ws.get_mlflow_tracking_uri()
mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
mlf_logger._run_id = run.id

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, help="data directory")
parser.add_argument('--epochs', type=int, help="epochs")
parser.add_argument('--batch_size', type=int, help="batch_size")
parser.add_argument('--drop_out', type=float, help="drop_out")
parser.add_argument('--hidden_dim', type=int, help="hidden_dim")
parser.add_argument('--layer_dim', type=int, help="layer_dim")
parser.add_argument('--embedding_dim', type=int, help="embedding_dim")
parser.add_argument('--vocab_size', type=int, help="vocab_size")
args = parser.parse_args()

print("Argument 1: %s" % args.datadir)
print("Argument 2: %s" % args.epochs)
print("Argument 3: %s" % args.batch_size)
print("Argument 4: %s" % args.drop_out)
print("Argument 5: %s" % args.hidden_dim)
print("Argument 6: %s" % args.layer_dim)
print("Argument 7: %s" % args.embedding_dim)
print("Argument 8: %s" % args.vocab_size)

npz = np.load(args.datadir)
x = npz['arr_0']
y = npz['arr_1']
print(x.shape)
print(y.shape)

x = torch.tensor(x, dtype=torch.int64)
y = torch.tensor(y, dtype=torch.int64)

dataset = torch.utils.data.TensorDataset(x, y)
num_train = int(len(dataset) * 0.6)
num_validation = int(len(dataset) * 0.2)
num_test = len(dataset) - num_train - num_validation

torch.manual_seed(0)
train, validation, test = torch.utils.data.random_split(dataset, [num_train, num_validation, num_test])

num_workers = 4

train_dataloader = DataLoader(train,      batch_size=args.batch_size, shuffle=True,  num_workers=num_workers)
val_dataloader   = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
test_dataloader  = DataLoader(test,       batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

torch.manual_seed(0)
net = LSTMModel(drop_out=args.drop_out, hidden_dim=args.hidden_dim, layer_dim=args.layer_dim, embedding_dim=args.embedding_dim, vocab_size=args.vocab_size)
trainer = Trainer(max_epochs=args.epochs, logger=mlf_logger, callbacks=[EarlyStopping(monitor='val_loss')])
trainer.fit(net, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

trainer.test(test_dataloaders=test_dataloader)
metrics = trainer.callback_metrics

print('val_loss', metrics['val_loss'].item())
print('val_acc', metrics['val_acc'].item())
print('test_loss', metrics['test_loss'].item())
print('test_acc', metrics['test_acc'].item())

os.makedirs('./outputs/models', exist_ok=True)
torch.save(net.state_dict(), './outputs/models/text_classifier_lstm.pt')
