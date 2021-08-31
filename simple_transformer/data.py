import torch
from torch.utils.data import Dataset

# spam detection
data = ['you won 20000 dollars',
                'How are you today?'
            , 'enter your credit card details asap for wining the big prize!',
                'please check this data scientist position']

labels = [1, 0, 1, 0]

class CustomSentimentDataset(Dataset):
    def __init__(self):
        self.data = data

        self.processed_data = []
        self.labels = labels

        # tokenize the sentences and create a vocab
        self.vocab = {}

        for example in self.data:
            idxs = []
            tokenized_example = example.strip().split()
            for token in tokenized_example:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                idxs.append(self.vocab[token])

            self.processed_data.append(torch.LongTensor(idxs))


    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        example = self.processed_data[idx]
        label = self.labels[idx]

        return example, label
