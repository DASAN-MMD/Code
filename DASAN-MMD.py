#https://github.com/easezyc/deep-transfer-learning/tree/master/UDA/pytorch1.0/DSAN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os,sys
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial
import copy
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time
import argparse
import seaborn as sns
import datetime
import os, sys
from matplotlib.pyplot import imshow, imsave
import distance_minimization  
from sklearn.metrics import f1_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = batch_size = 10
GAMMA = 10 ^ 3
LAMBDA = 0.25

classes = ('bending', 'falling', 'lie_down', 'running',
           'sit_down', 'stand_up', 'walking')

train_dataset_path="./Dataset_S1_S2_S3/training1/training"
test_dataset_path="./Dataset_S1_S2_S3/training2/training"

mean=[0.7290,0.8188,0.6578]
std=[0.2965,0.1467,0.2864]

train_transforms=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean),torch.Tensor(std))
])

train_dataset=torchvision.datasets.ImageFolder(root=train_dataset_path,transform=train_transforms)
test_dataset=torchvision.datasets.ImageFolder(root=test_dataset_path,transform=train_transforms)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=10,shuffle=True)

#Test data spliting
val_pct=0.8
rand_seed=42
n_val=int(val_pct*len(test_dataset))
np.random.seed(rand_seed)
idxs=np.random.permutation(len(test_dataset))


test_train_sampler=SubsetRandomSampler(idxs[:n_val])
test_train_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=test_train_sampler) #Use for model training with 0.8 per target samples

test_sampler=SubsetRandomSampler(idxs[n_val:])
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0, sampler=test_sampler) # Use for model testing with 0.2 per target samples

class FeatureExtractor(nn.Module):
    """
        Feature Extractor
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(3,2)
        self.dropout = nn.Dropout(.3)
        self.relu = nn.ReLU()


    def forward(self, src_features, tgt_features):
        src_features = self.relu(self.bn1(self.conv1(src_features)))
        src_features = self.pool(src_features)
        src_features = self.dropout(src_features)
        src_features = self.relu(self.bn1(self.conv2(src_features)))
        src_features = self.pool(src_features)
        src_features = self.dropout(src_features)
        src_features = self.relu(self.bn2(self.conv3(src_features)))
        src_features = self.pool(src_features)
        src_features = self.dropout(src_features)
        src_features = self.relu(self.conv4(src_features))
        src_features = src_features.view(-1, 256)


        tgt_features = self.relu(self.bn1(self.conv1(tgt_features)))
        tgt_features = self.pool(tgt_features)
        tgt_features = self.dropout(tgt_features)
        tgt_features = self.relu(self.bn1(self.conv2(tgt_features)))
        tgt_features = self.pool(tgt_features)
        tgt_features = self.dropout(tgt_features)
        tgt_features = self.relu(self.bn2(self.conv3(tgt_features)))
        tgt_features = self.pool(tgt_features)
        tgt_features = self.dropout(tgt_features)
        tgt_features = self.relu(self.conv4(tgt_features))
        tgt_features = tgt_features.view(-1, 256)

        return src_features, tgt_features

class Classifier(nn.Module):
    """
        Classifier
    """
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(256, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, 7)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src_features, tgt_features, src_label):
        src_features = self.fc1(src_features)
        src_features = self.relu(src_features)
        src_features = self.fc2(src_features)
        src_features = self.relu(src_features)
        src_pred = self.fc3(src_features)
        src_pred = self.softmax(src_pred)

        tgt_features = self.fc1(tgt_features)
        tgt_features = self.relu(tgt_features)
        tgt_features = self.fc2(tgt_features)
        tgt_features = self.relu(tgt_features)
        tgt_pred = self.fc3(tgt_features)
        tgt_pred = self.sf(tgt_pred)

        #loss_lmmd = self.lmmd_loss.get_loss(s1, t1, s_label, t_pred)
        return src_pred, src_features, tgt_features, tgt_pred

    def predict(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class Discriminator(nn.Module):
    """
        Simple Discriminator 
    """

    def __init__(self, latent_size=256, num_classes=1):
        super(Discriminator, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

Feats = FeatureExtractor().to(device)
Cls = Classifier().to(device)
Dis = Discriminator().to(device)


F_opt = torch.optim.Adam(Feats.parameters(),lr=0.0001)
C_opt = torch.optim.SGD(Cls.parameters(),lr=0.001)
D_opt = torch.optim.SGD(Dis.parameters(),lr=0.001)

bce = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

max_epoch = 30
step = 0
n_batches = len(test_train_loader)//batch_size

D_src = torch.ones(batch_size, 1).to(device) 
D_tgt = torch.zeros(batch_size, 1).to(device) 
D_labels = torch.cat([D_src, D_tgt], dim=0)


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

def mmd_loss(x_src, x_tar):
    return distance_minimization.mmd_loss(x_src, x_tar, x, y, kernel_bandwidth=1.0)


test_loader_set = iter(test_train_loader)

def sample_mnist(step, n_batches):
    global test_loader_set
    if step % n_batches == 0:
        test_loader_set = iter(test_train_loader)
    return test_loader_set.next()

D_train_losses = []
F_train_losses = []
C_train_losses = []
total_train_losses = []
model_train_accuracy = []
patience, trials, min_loss = 5, 0, 1000

training_start_time = time.time()
for epoch in range(100):
    size = 0
    correct = 0
    D_running_loss = 0.0
    F_running_loss = 0.0
    C_running_loss = 0.0
    Ltot_running_loss = 0.0
    Feats.train()
    Cls.train()
    Dis.train()
    for step, (images_s, labels_s) in enumerate(train_loader):  # loop over the dataset multiple times
        images_t, labels_t = sample_mnist(step, n_batches)
        images_s, labels_s, images_t, labels_t = images_s.to(device), labels_s.to(device), images_t.to(device), labels_t.to(device)

        # zero gradients for optimizer
        F_opt.zero_grad()
        C_opt.zero_grad()

        # compute loss for critic
        images_src, images_tgt = Feats(images_s, images_t)
        src_pred, src_features, tgt_features, tgt_pred = Cls(images_src, images_tgt, labels_s)
        loss_mmd = mmd_loss(src_features, tgt_features)

        loss_src = criterion(src_pred, labels_s)
        loss_tgt = criterion(tgt_pred, labels_s)
        loss = loss_src + 0.01 * loss_tgt + (LAMBDA * loss_mmd)

        loss.backward()
        F_opt.step()
        C_opt.step()
        F_running_loss += loss.item()  

        images_src, images_tgt = Feats(images_s,images_t)
        h= torch.cat([images_src,images_tgt], dim=0)
        y = Dis(h.detach())

        Ld = bce(y, D_labels)
        D_opt.zero_grad()
        Ld.backward()
        D_opt.step()

        src_pred, src_features, tgt_features, tgt_pred = C(images_src,images_tgt,labels_s)
        y = Dis(h)
        Lc = criterion(src_pred, labels_s)
        Ld = bce(y, D_labels)
        Ltot = Lc - LAMBDA * Ld

        F_opt.zero_grad()
        C_opt.zero_grad()
        D_opt.zero_grad()

        Ltot.backward()

        C_opt.step()
        F_opt.step()

        D_running_loss += Ld.item() 
        C_running_loss += Lc.item() 
        Ltot_running_loss += Ltot.item()  

        images_src, images_tgt = F(images_s, images_t)
        C_predict_tgt = C.predict(images_tgt)
        _, predicted = torch.max(C_predict_tgt, 1)
        correct += (predicted == labels_t).sum().item()
        size += labels_t.size(0)
        step += 1

    F_running_loss = F_running_loss / (len(train_loader))
    D_running_loss = D_running_loss / (len(test_train_loader))
    C_running_loss = C_running_loss / (len(test_train_loader))
    Ltot_running_loss = Ltot_running_loss / (len(train_loader))
    F_train_losses.append(F_running_loss)
    D_train_losses.append(D_running_loss)
    C_train_losses.append(C_running_loss)
    total_train_losses.append(Ltot_running_loss)
    tr_accuracy = float(correct / size)
    model_train_accuracy.append(tr_accuracy * 100)

    if F_running_loss < min_loss:
        trials = 0
        best_acc = F_running_loss
        epoch_num_Encoder = epoch +1
    else:
        trials += 1
        epoch_num_Encoder = epoch + 1
        if trials >= patience:
            break

training_time = time.time() - training_start_time
training_time = training_time/60
print('Training_Time:{:.2f}'.format(training_time))
print('Finished Training')

with torch.no_grad():
    correct = 0
    size = 0
    all_preds = torch.tensor([]).to(device)
    all_labels = torch.tensor([]).to(device)
    accuracyT = []
    for idx, (tgt, labels) in enumerate(test_loader):
        tgt, labels = tgt.to(device), labels.to(device)
        all_labels = torch.cat((all_labels, labels), dim=0)  
        _,images_tgt = Feats(tgt,tgt)
        C_predict_tgt = Cls.predict(images_tgt)
        _, predicted = torch.max(C_predict_tgt, 1)
        all_preds = torch.cat((all_preds, predicted), dim=0) 
        correct = (predicted == labels).sum().item()
        size = labels.size(0)
    accuracy = float(correct / size)
    
print('f1_score(micro):{:.2f}'.format(f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='micro')))
print('f1_score(macro):{:.2f}'.format(f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')))
