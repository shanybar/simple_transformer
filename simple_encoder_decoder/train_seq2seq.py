import argparse
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from simple_encoder_decoder.seq2seq_data import ReverseStringDataset
from simple_encoder_decoder.encoder_decoder import EncoderDecoder

parser = argparse.ArgumentParser(description='Training a seq2seq model')

parser.add_argument('--epochs', type=int,
                    help=' number of epochs')
parser.add_argument('--lr', type=float,
                    help=' learning rate')

args = parser.parse_args()

def train():
    epochs=args.epochs
    lr=args.lr

    training_data = ReverseStringDataset()
    train_dataloader = DataLoader(training_data,batch_size=1, shuffle=True)
    net = EncoderDecoder(embed_dim=50, hid_dim= 50,vocab_len=training_data.vocab_len)
    optim = SGD(net.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    for epoch in range(epochs):
        epoch_loss = 0.

        for step, (encoder_inp, decoder_inp, decoder_trg) in enumerate(train_dataloader):
            pred = net(encoder_inp.squeeze(), decoder_inp.squeeze())
            batch_loss = loss_fn(pred, decoder_trg.view(-1))
            epoch_loss += batch_loss

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

        print('epoch %d loss %f\n' % (epoch, epoch_loss))



if __name__ == "__main__":
    train()