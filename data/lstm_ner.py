import codecs
import torch
import torch.nn as nn
from utils import create_dico, create_mapping, zero_digits
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import random

LENGTH = 15

def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    label = []
    labels = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) >= 2:
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word[0])
            label.append(word[3])
    if len(sentence) >= 2:
        sentences.append(sentence)
        labels.append(label)

    return sentences,labels

def word2id(list):
    wordid = []
    for word in list:
        if word not in word_to_ix:
             word_to_ix[word] = len(word_to_ix)
        else:
            wordid.append(word_to_ix[word])
    return wordid
    #return torch.LongTensor(np.array(wordid)).unsqueeze(1)

def label2id(list):
    labelid = []
    for label in list:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
        else:
            labelid.append(label_to_ix[label])

    return torch.LongTensor(np.array(labelid)).unsqueeze(1)

def Pad(list):
    if len(list) > LENGTH  :
        return list[:LENGTH]
    else:
        while len(list) != LENGTH:
            list.append('PAD')
    return list

def label2onehot(y):
    y_onehot = torch.FloatTensor(len(y), 9)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot

def init():
    global sentences, labels
    sentences, labels = load_sentences("data/eng.train", None)
    for sentence in sentences:
        word2id(sentence)
    for label in labels:
        label2id(label)

def display_labelid(len,pred_y):
    my_answer = []
    for i in pred_y:
        for word in label_to_ix:
            if i == label_to_ix[word]:
                my_answer.append(word)
    return my_answer[:len]


word_to_ix = {}
word_to_ix['PAD'] = 0
label_to_ix = {}
label_to_ix['PAD'] = 0
sentences = []
labels = []
init()
#
# #print(torch.zeros(len(label_to_ix[0]),9).scatter_(1,label_to_ix[0],1))
# print(word_to_ix)
# print(label_to_ix)
#
# sentence = Pad(sentences[0],15)
# label    = Pad(labels[0],15)
#
# sentenceid = word2id(sentence)
# sentenceid = torch.LongTensor(np.array(sentenceid)).unsqueeze(1)
# labelbatch = label2id(labels[0])
# y_onehot = label2onehot(labelbatch)
# print(labelbatch)
# print(y_onehot)
#
# sentenceid = Variable(sentenceid)
# embeds = nn.Embedding(len(word_to_ix),20)
# hello_embed = embeds(sentenceid)
#
# for i,item in enumerate(hello_embed):
#     print(item)
#     print(sentence[i])
#     print("-------------------------")
# #print(hello_embed)


def random_batch():
    return random.sample(list(zip(sentences[:10000],labels[:10000])),50)


EPOCH = 1
BATCH_SIZE = 64
WORD_LENGTH = 15
LR = 0.01



class RNN(nn.Module):
    def __init__(self, input_size,n_dim,hidden_size,n_layers=2):
        super(RNN,self).__init__()
        self.embedding=nn.Embedding(input_size, n_dim)
        self.rnn = nn.LSTM(
            input_size=n_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )

        self.out = nn.Linear(hidden_size,9)

    def forward(self,x):
        emb = self.embedding(x)
        r_out,(h_n,h_c) = self.rnn(emb,None)
        a = Variable(torch.FloatTensor(LENGTH, 9))
        #print(x.shape,emb.shape,r_out.shape)
        for i in range(LENGTH):
            a[i, :] = F.softmax(self.out(r_out[:, i, :]), dim=1)

        # out = self.out(r_out[:,-1,:])
        #print(r_out.shape,h_c.shape,h_n.shape,r_out[:,-1,:].shape)
        # out = F.softmax(r_out)
        #print(a)
        return a


model = RNN(len(word_to_ix),100,64,1)
# print(model)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()
#
# print(len(sentences))
# for epoch in range(1000):
#     print(str(epoch)+" : -------------------------------------------------"+str(epoch)+"---------------------")
#     for step,(x,y) in enumerate(random_batch()):
#         b_x = Variable(torch.LongTensor(np.array(word2id(Pad(x,15)))).unsqueeze(0))
#         b_y = Variable(label2id(Pad(y, 15)))
#         output = model(b_x)
#         loss = 0
#         for i in range(15):
#             #print(loss_func(output[i,:].unsqueeze(0), b_y[i,:]))
#             loss = loss + loss_func(output[i,:].unsqueeze(0), b_y[i,:])
#         print(loss)
#         # #     print(str(i)+"--------------------------")
#         optimizer.zero_grad()  # clear gradients for this training step
#         loss.backward()  # backpropagation, compute gradients
#         optimizer.step()
# torch.save(model,"/home/zxd/model")

model = torch.load("/home/zxd/model")
right = 0
total = 0

sens,labes = load_sentences("data/eng.train54019",None)
for step,(x,y) in enumerate(zip(sens,labes)):
#for step,(x,y) in enumerate(zip(sentences[10001:],labels[10001:])):
    length = len(x)
    b_x = Variable(torch.LongTensor(np.array(word2id(Pad(x)))).unsqueeze(0))
    b_y = label2id(Pad(y))
    output = model(b_x)
    pred_y = torch.max(output,1)[1].data.numpy().squeeze()
    y_answer = np.array(b_y).reshape(-1)

    for i,t in enumerate(y_answer):
        if pred_y[i] == y_answer[i] & y_answer[i] != 0:
            right +=1
    total += len(y)

    my_answer = display_labelid(length,pred_y)

    if length < LENGTH:
        print(x[:length])
        print("------------right answer---------------")
        print(y[:length])
        print("------------my answer----------------")

    else:
        print(x[:LENGTH])
        print("------------right answer---------------")
        print(y[:LENGTH])
        print("------------my answer----------------")
    print(my_answer)

    print("=============================================================")
print(right/float(total))












