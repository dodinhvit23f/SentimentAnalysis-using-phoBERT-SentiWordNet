import io
import glob, os
import csv
import random

def GetVec (Senti, s):
    #global Senti
    pos, neg, c = 0, 0, 0
    for w in s.split():
        if w in Senti:
            pos = pos+ Senti[w][0]
            neg = pos+ Senti[w][1]
            c=c+1
    if c!=0:
        pos= round(pos/c, 4)  # pos= pos/len(s) pos= pos/c
        neg= round(neg/c, 4)  #neg = neg/len(s) neg= neg/c
    return pos, neg

def GetVec1 (Senti, s):
    #global Senti
    pos, neg, c = 0,0, 0
    for w in s.split():
        if w in Senti:
            if Senti[w][0]>=Senti[w][1]:
                pos = pos+ Senti[w][0]
            else:
                neg = neg+ Senti[w][1]
    if len(s)!=0:
        return pos/len(s), neg/len(s)
    else:
        return 0, 0


def GetVec2 (Senti, s):
    #global Senti
    pos, neg, c = 0, 0, 0
    for w in s.split():
        if w in Senti:
            if Senti[w][0]>=Senti[w][1]:
                pos = pos + Senti[w][0]
            else:
                neg = neg + Senti[w][1]
    return pos, neg 


def GetVec3 (Senti, s):
    #global Senti
    pos, neg, c = 0,0, 0
    for w in s.split():
        if w in Senti:
            if Senti[w][0]>=Senti[w][1]:
                pos = max(pos, Senti[w][0])
               
            else:
                neg = max(neg , Senti[w][1])
    return pos, neg 

def getSentiWN(path):
    Senti =  dict({})
    f = open(path, 'r' , encoding="utf-8")
    for i in f:
        if i.strip()=="":
            continue
        ws= i.split('\t')
        key = ws[4].replace("#","")
        for i in ['0','1','2','3','4','5','6','7','8','9']:
            key= key.replace(i,"")
            pos= float(ws[2])
            neg=float(ws[3])
            if key not in Senti:
                Senti.update({key:[pos, neg]})
    f.close()
    return Senti
 

