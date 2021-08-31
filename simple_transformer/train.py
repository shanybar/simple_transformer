import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from simple_transformer.simple_transformer_encoder import SelfAttention
from simple_transformer.data import CustomSentimentDataset

parser = argparse.ArgumentParser(description='Training a classifier')

parser.add_argument('--epochs', type=int,
                    help=' number of epochs')
parser.add_argument('--lr', type=float,
                    help=' learning rate')

args = parser.parse_args()

def train():
    epochs = args.epochs
    lr = args.lr
    training_data = CustomSentimentDataset()
    net = SelfAttention(d_embed=30, d_output=2, len_vocab=len(training_data.vocab))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), lr=lr)

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    for epoch in range(epochs):
        ep_loss = 0.
        for step, (example, label) in enumerate(train_dataloader):
            pred = net(example.squeeze())
            # label =label.unsqueeze(0)
            batch_loss = loss_fn(pred, label)
            batch_loss.backward() # do backprop
            optim.step() # update parameters
            optim.zero_grad() # reset parameters
            ep_loss += batch_loss

        print(epoch, ep_loss)



if __name__ == "__main__":
    train()