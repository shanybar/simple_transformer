import torch
from torch.utils.data import Dataset

class ReverseStringDataset(Dataset):
    def __init__(self):

        self.vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, '<sos>': 5, '<eos>': 6}
        self.vocab_len = len(self.vocab)
        self.idx_to_vocab = dict((v, k) for k, v in self.vocab.items())
        self.train_seq_len = 10
        self.num_train_examples = 5

        self.train_encoder_inputs = None
        self.train_decoder_inputs = None
        self.train_decoder_targets = None

        self.generate_seq2seq_data()

    def __len__(self):
        return self.num_train_examples

    def __getitem__(self, idx):
        encoder_inp = self.train_encoder_inputs[idx]
        decoder_inp = self.train_decoder_inputs[idx]
        decoder_trg = self.train_decoder_targets[idx]

        return encoder_inp, decoder_inp, decoder_trg

    def generate_seq2seq_data(self):
        # Generate toy data
        train_inputs = torch.LongTensor(self.num_train_examples,
                                        self.train_seq_len).random_(0, len(self.vocab)-2)
        inv_idx = torch.arange(self.train_seq_len-1, -1, -1).long()
        train_targets = train_inputs[:, inv_idx] # targets are just reverse of the inputs
        sos_vec = torch.LongTensor(self.num_train_examples, 1)
        sos_vec[:] = self.vocab['<sos>']
        eos_vec = torch.LongTensor(self.num_train_examples, 1)
        eos_vec[:] = self.vocab['<eos>']
        self.train_encoder_inputs = torch.cat((train_inputs, eos_vec), dim=1)
        self.train_decoder_inputs = torch.cat((sos_vec, train_targets), dim=1)
        self.train_decoder_targets = torch.cat((train_targets, eos_vec), dim=1)

        print('encoder input :', ' '.join([self.idx_to_vocab[w] for w
                                           in self.train_encoder_inputs[0].numpy()]))
        print('decoder input:', ' '.join([self.idx_to_vocab[w] for w
                                          in self.train_decoder_inputs[0].numpy()]))
        print('decoder target:', ' '.join([self.idx_to_vocab[w] for w
                                           in self.train_decoder_targets[0].numpy()]))



