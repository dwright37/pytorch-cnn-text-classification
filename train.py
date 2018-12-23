import random
import logging
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from model import CharacterCNN
from model import weights_init
from model import DatasetReader

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
th.manual_seed(manualSeed)

log = logging.getLogger("cnn_train")
log.setLevel(logging.INFO)
log.addHandler(logging.FileHandler('train.log'))
#Character vocab
char_inventory = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%ˆ&*˜`+-=<>()[]{}\n'

vocab = {c:i for i,c in enumerate(char_inventory)}

# One hot embedding size
nchars = len(char_inventory)

#Input length
input_length = 1014

# Decide which device we want to run on
ngpu = 1
device = th.device("cuda:0" if (th.cuda.is_available() and ngpu > 0) else "cpu")

#Batch size
batch_size = 128

#Number of threads for the data loader
workers = 2

#Number of epochs
nepochs = 20

cnn = CharacterCNN(nchars).to(device)
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
cnn.apply(weights_init)
cnn.train()

train_data = DatasetReader('/projects/ibm_aihl/dwright/datasets/cnn_text_classification_data/yelp_review_full_csv/train.csv', vocab, input_length, nchars)
dataloader = th.utils.data.DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


# The objective
criterion = nn.CrossEntropyLoss()

#Optimizer
optimizer = optim.Adam(cnn.parameters())

losses = []

#Main training loop
for epoch in range(nepochs):
    for i, batch in enumerate(dataloader):
        inputs = batch[0].type(th.FloatTensor).to(device)
        targets = batch[1].type(th.LongTensor).to(device)
        
        optimizer.zero_grad()
        preds = cnn(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        # Output training stats
        if i % 5 == 0:
            log.info('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch, nepochs, i, len(dataloader), loss.item()))
            losses.append(loss)
        
        if i % 100 == 0:
            th.save({
                'model': cnn.state_dict(),
                'epoch': epoch,
            }, 'cnn.pth')
th.save({
    'model': cnn.state_dict(),
    'epoch': epoch,
}, 'cnn.pth')
