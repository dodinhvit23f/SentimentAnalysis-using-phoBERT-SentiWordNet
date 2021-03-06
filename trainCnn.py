import pdb

import pandas as pd
from models import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import torch
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import argparse
from CNNModel import SemtimentNetwork
from transformers.modeling_utils import * 
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
from utils import *
from Extract_SentiVec import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='./data/train_all.csv')
parser.add_argument('--dict_path', type=str, default="./phobert/dict.txt")
parser.add_argument('--config_path', type=str, default="./phobert/config.json")
parser.add_argument('--rdrsegmenter_path', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default='./phobert/model.bin')
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--accumulation_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=8)
#parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=69)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--ckpt_path', type=str, default='./models')
parser.add_argument('--bpe-codes', default="./phobert/bpe.codes",type=str, help='path to fastBPE BPE')

args = parser.parse_args()
bpe = fastBPE(args)
rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m') 

seed_everything(69)

# Load model
config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    num_labels=1
)

model_bert = RobertaForAIViVN.from_pretrained(args.pretrained_path, config=config) #args.pretrained_path  ???????ng d???n t???i pretrained Bert 
model_bert.cuda() #????a l??n Cuda

SemtimentNet = SemtimentNetwork(2)
SemtimentNet.load_state_dict(torch.load("./models/Cnn.bin"))
SemtimentNet.cuda()

if torch.cuda.device_count(): # N???u c?? cuda
    print(f"Training using {torch.cuda.device_count()} gpus")
    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta
else:
    tsfm = model_bert.roberta

# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file(args.dict_path)

# Load training data
train_df = pd.read_csv(args.train_path,sep='\t').fillna("###")
train_df.text = train_df.text.progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))

y = train_df.label.values
#senti = train_df.label.values # truong senti
#senti = np.ones((len(train_df.label.values),768)) # truong senti
#load sentiwordnet
SentiWN = getSentiWN("VietSentiWordnet_Ver1.3.5.txt")
senti1, senti3=[], []
for s in train_df.text:
    pos, neg = GetVec3(SentiWN, s)
    temp =[pos,]*384+[neg,]*384
    senti1.append(temp)

    pos, neg = GetVec(SentiWN, s)
    temp =[pos,]*384+[neg,]*384
    senti3.append(temp)

senti1 = np.array(senti1)
senti3 = np.array(senti3)
    
    
# Thay ?????i m???i ??? ????y s??? ch??? ch???a data sau ???? theo batch chuy???n sang d???ng c???a
# X_train = convert_lines(train_df, vocab, bpe,args.max_sequence_length) # chuyen X_train th??nh ma tran, moi d??ng l?? danh sach c??c id
X_train = convert_lines_cnn(train_df, vocab, bpe,1000)
# Creating optimizer and lr schedulers
param_optimizer = list(model_bert.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = int(args.epochs*len(train_df)/args.batch_size/args.accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler

if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)

#pdb.set_trace()

#splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train,  y))  # chia lam 5 phan
splits = list(StratifiedKFold(n_splits=20, shuffle=True, random_state=123).split(X_train,  y))  # chia lam 20 phan

# Chia vector senti

for  fold, (train_idx, val_idx) in enumerate(splits):

    print("Training for fold {}".format(fold))
    best_score = 0
    if fold != args.fold:
        continue
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long), torch.tensor(y[train_idx],dtype=torch.long), torch.tensor(senti1[train_idx],dtype=torch.long))
     
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(y[val_idx],dtype=torch.long), torch.tensor(senti1[val_idx],dtype=torch.long))
    

    tq = tqdm(range(args.epochs + 1))
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = False
    frozen = True
    for epoch in tq:

        if epoch > 0 and frozen:
            for child in tsfm.children():
                for param in child.parameters():
                    param.requires_grad = True
            frozen = False
            del scheduler0
            torch.cuda.empty_cache()

        val_preds = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        avg_loss = 0.
        avg_accuracy = 0.

        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        for i,(x_batch, y_batch, senti1_batch) in pbar:
            model_bert.train()

            cnn_data = torch.unsqueeze(x_batch, 1)
            size = len(cnn_data)
            cnn_vec = SemtimentNet(cnn_data.cuda().float(), size)
            cnn_vec = cnn_vec.long()

            input = np.zeros((size,  args.max_sequence_length))

            for idx in range(size):
                input[idx, :] = x_batch[idx, :256]
            input = torch.from_numpy(input).long()
            #pdb.set_trace()
            y_pred = model_bert(input.cuda(), senti1_batch.cuda(), attention_mask=(input > 0).cuda(), cnn_vector=cnn_vec)

            #y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            loss =  F.binary_cross_entropy_with_logits(y_pred.view(-1).cuda(),y_batch.float().cuda())
            loss = loss.mean()
            loss.backward()
            if i % args.accumulation_steps == 0 or i == len(pbar) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()
            lossf = loss.item()
            pbar.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)

        model_bert.eval()

        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        for i,(x_batch, y_batch, senti1_batch) in pbar:


            cnn_data = torch.unsqueeze(x_batch, 1)
            size = len(cnn_data)
            cnn_vec = SemtimentNet(cnn_data.cuda().float(), size)
            cnn_vec = cnn_vec.long()
            input = np.zeros((size, args.max_sequence_length) )

            for idx in range (size):
                input[idx, :] = x_batch[idx, :256]
            input = torch.from_numpy(input).long()

            y_pred = model_bert(input.cuda(), senti1_batch.cuda(),attention_mask=(input>0).cuda(), cnn_vector = cnn_vec )
            #y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            y_pred = y_pred.squeeze().detach().cpu().numpy()
            val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])
        val_preds = sigmoid(val_preds)

        best_th = 0
        score = f1_score(y[val_idx], val_preds > 0.5)
        score_acc = accuracy_score(y[val_idx], val_preds > 0.5)
        print(f"\nAUC = {roc_auc_score(y[val_idx], val_preds):.4f}, F1 score @0.5 = {score:.4f}, accuracy score = {score_acc:.4f}")
        if score >= best_score:
            torch.save(model_bert.state_dict(),os.path.join(args.ckpt_path, f"model_senti_cnn_{fold}.bin"))
            best_score = score

            
# danh gia mo h??nh

test_path = "./data/test-vivn.csv"
test_df = pd.read_csv(test_path,sep='\t').fillna("###")
test_df.text = test_df.text.progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))

y = test_df.label.values

#load sentiwordnet
senti1, senti3=[], []
for s in test_df.text:
    pos, neg = GetVec3(SentiWN, s)
    temp =[pos,]*384+[neg,]*384
    senti1.append(temp)

    pos, neg = GetVec(SentiWN, s)
    temp =[pos,]*384+[neg,]*384
    senti3.append(temp)

senti1 = np.array(senti1)
senti3 = np.array(senti3)
    
X_test = convert_lines_cnn(test_df, vocab, bpe,1000) # chuyen X_train th??nh ma tran, moi d??ng l?? danh sach c??c id
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test,dtype=torch.long), torch.tensor(y,dtype=torch.long), torch.tensor(senti1, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

test_preds =[]
model_bert.eval()
pbar = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
for i,(x_batch, y_batch, senti1_batch) in pbar:

    cnn_data = torch.unsqueeze(x_batch, 1)
    size = len(cnn_data)
    cnn_vec = SemtimentNet(cnn_data.cuda().float(), size)
    cnn_vec = cnn_vec.long()

    input = np.zeros((size, args.max_sequence_length ))

    for idx in range(size):
        input[idx, :] = x_batch[idx, :256]
    input = torch.from_numpy(input).long()
    # pdb.set_trace()
    y_pred = model_bert(input.cuda(), senti1_batch.cuda(), attention_mask=(input > 0).cuda(), cnn_vector=cnn_vec)
    #y_pred = model_bert(x_batch.cuda(),  attention_mask=(x_batch>0).cuda())
    y_pred = y_pred.squeeze().detach().cpu().numpy()
    test_preds = np.atleast_1d(y_pred) if test_preds is None else np.concatenate([test_preds, np.atleast_1d(y_pred)])
test_preds = sigmoid(test_preds)

print("test_preds: ", len(test_preds), "\t  y:", len(y))

f1_score = f1_score(y, test_preds > 0.5)
score_acc = accuracy_score(y, test_preds > 0.5)
precision_score = precision_score(y, test_preds > 0.5)
recall_score = recall_score(y, test_preds > 0.5)
f=open("test-vivn.txt", "w")
f.write("VIVN using MLP \n")
f.write("accuracy_score: "+str(score_acc)+" \t  f1_score: "+str(f1_score) +"\t recall_score: "+str(recall_score) +"\t  precision_score: "+str(precision_score))
f.close()

'''
x_batch = torch.tensor(X_test,dtype=torch.long)
y_batch=  torch.tensor(y,dtype=torch.long)
senti_batch  = torch.tensor(senti,dtype=torch.long)
#model_bert.eval()

y_pred = model_bert(x_batch.cuda(), senti_batch.cuda(), attention_mask=(x_batch>0).cuda())
y_pred = y_pred.squeeze().detach().cpu().numpy()
test_preds = sigmoid(test_preds)

print("test_preds ", len(test_preds), " Y: ", len(y))
'''



