import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
#CNN
class CharacterCNN(nn.Module):
    def __init__(self, nchars):
        super(CharacterCNN, self).__init__()
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv1d(nchars, 256, 7),
            nn.MaxPool1d(3),
            nn.ReLU(),
            #Layer 2
            nn.Conv1d(256, 256, 7),
            nn.MaxPool1d(3),
            nn.ReLU(),
            #Layer 3
            nn.Conv1d(256, 256, 3),
            nn.ReLU(),
            #Layer 4
            nn.Conv1d(256, 256, 3),
            nn.ReLU(),
            #Layer 5
            nn.Conv1d(256, 256, 3),
            nn.ReLU(),
            #Layer 6
            nn.Conv1d(256, 256, 3),
            nn.MaxPool1d(3),
            nn.ReLU(),
            #Flatten
            Flatten(),
            #Fully connected 1
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #Fully connected 2
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            #Output layer for yelp reviews
            nn.Linear(1024, 5)
        )
        
    def forward(self, input):
        return self.net(input)

class DatasetReader(Dataset):
    def __init__(self, filename, vocab, input_length, nchars):
        self.data = pd.read_csv(filename, sep=",", quotechar='"', header=None)
        self.vocab = vocab
        self.input_length = input_length
        self.nchars = nchars

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        vocab = self.vocab
        input_length = self.input_length
        nchars = self.nchars
        row = self.data.values[i]
        
        target = row[0] - 1
        text = row[1][:input_length].lower()

        #Normalize the text size
        if len(text) < input_length:
            text = text + ' '*(input_length - len(text))

        onehots = []
        for c in text:
            onehot = np.zeros(nchars)
            if c in vocab:
                onehot[vocab[c]] = 1.0
            onehots.append(onehot)

        return np.asarray(onehots).T, target
    
    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        
        return np.vstack(inputs), np.vstack(targets)
