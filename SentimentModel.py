import pdb

import torch
import torch.nn as nn
import  torch.nn.functional as F

from vncorenlp import VnCoreNLP
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
import argparse
from transformers import *
from utils import *

from transformers.modeling_utils import *

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """

    string = string.replace("\\", "")
    # string = bytes(string, 'ascii','ignore')
    string = bytes(string, 'utf-8', 'ignore')
    string = re.sub(b'\'', b'', string)
    string = re.sub(b'\"', b'', string)
    return string.strip().lower()

def getModel():
    third_layer = nn.Sequential(nn.Conv1d(128, 1, kernel_size=5), nn.ReLU())
    frist_layer = nn.Sequential(nn.Conv1d(128, 1, kernel_size=3), nn.ReLU())
    second_layer = nn.Sequential(nn.Conv1d(128, 1, kernel_size=4), nn.ReLU())

    print(third_layer)
    merge_layer = nn.Sequential(frist_layer, second_layer, third_layer)
    merge_layer = nn.Sequential( merge_layer,
                                       nn.Conv1d(128,1, kernel_size=5),
                                       nn.ReLU(),
                                       nn.MaxPool1d(5),
                                       nn.Conv1d(128, 3, kernel_size=5),
                                        nn.ReLU(),
                                        nn.MaxPool1d(5),
                                       nn.Flatten(),
                                       nn.Linear(128, 128),
                                       nn.ReLU(),
                                       nn.Softmax(dim=2)
                                       )

    return merge_layer
class SemtimentNetwork(nn.Module):
    def __init__(self, number_of_class):
        super(SemtimentNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=4, stride=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(in_channels=6, out_channels=12, kernel_size=5, stride=1)
        self.class_number = number_of_class

        self.fc1 = nn.Linear( in_features=12*4*4, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=self.class_number)

    def forward(self, x):
        #input layer
        x = x
        #first hidden layer
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = F.max_pool1d(x1, kernel_size =2)

        x2 = self.conv2(x)
        x2 = F.relu(x2)
        x2 = F.max_pool1d(x2, kernel_size =2)

        x3 = self.conv3(x)
        x3 = F.relu(x3)
        x3 = F.max_pool1d(x3, kernel_size =2)

        # concatenate layer
        x = torch.cat((x,x1,x3), dim=0)
        # second hidden layer
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=5)
        # third hidden layer
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=30)

        #x = torch.flatten(x)
        x = x.reshape(-1, 12*4*4)
        x = self.fc1(x)
        out = self.out(x)

        return out



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_path', type=str, default='./data/train.csv')
    parser.add_argument('--dict_path', type=str, default="./phobert/dict.txt")
    parser.add_argument('--config_path', type=str, default="./phobert/config.json")
    parser.add_argument('--rdrsegmenter_path', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, default='./phobert/model.bin')
    parser.add_argument('--max_sequence_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--ckpt_path', type=str, default='./models')
    parser.add_argument('--bpe-codes', default="./phobert/bpe.codes", type=str, help='path to fastBPE BPE')

    args = parser.parse_args()

    rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m')

    model = SemtimentNetwork(2)
    print(model)
    pdb.set_trace()

    vocab.add_from_file(args.dict_path)

    train_df = pd.read_csv(args.train_path, sep='\t').fillna("###")
    train_df.text = train_df.text.progress_apply(
        lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
    y = train_df.label.values
    X_train = convert_lines(train_df, vocab, bpe, args.max_sequence_length)


    """
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(0, 2):
    # get the inputs; data is a list of [inputs, labels]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(torch.from_numpy(x_train))
    loss = criterion(outputs, torch.from_numpy(y_train))
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:  # print every 2000 mini-batches
       print('[%d, %5d] loss: %.3f' %
             (epoch + 1, i + 1, running_loss / 2000))
       running_loss = 0.0
    """