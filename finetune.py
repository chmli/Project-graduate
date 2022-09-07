import torch
import torch.nn as nn
import seaborn
seaborn.set()
import numpy as np
import torch.functional as F
def pp():
    print("R")
class FullyConnected(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim,par):
        super(FullyConnected,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hid_dim1),nn.BatchNorm1d(hid_dim1),nn.PReLU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim1,hid_dim2),nn.BatchNorm1d(hid_dim2),nn.PReLU())
        self.layer3 = nn.Sequential(nn.Linear(hid_dim2,out_dim)) 
        self.coeff = nn.Parameter(torch.tensor([par]))

    def forward(self,X):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.coeff*x + (1-self.coeff)*X
        return x

class FullyConnectedApp(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim):
        super(FullyConnectedApp,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hid_dim1),nn.BatchNorm1d(hid_dim1),nn.PReLU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim1,hid_dim2),nn.BatchNorm1d(hid_dim2),nn.PReLU())
        self.layer3 = nn.Sequential(nn.Linear(hid_dim2,out_dim)) 
        #self.coeff = nn.Parameter(torch.tensor([par]))

    def forward(self,X):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.cat((X,x),dim=1)
        #x = self.coeff*x + (1-self.coeff)*X
        return x

class FullyConnected_dropout(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim,par,dp1,dp2):
        super(FullyConnected_dropout,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hid_dim1),nn.BatchNorm1d(hid_dim1),nn.PReLU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim1,hid_dim2),nn.BatchNorm1d(hid_dim2),nn.PReLU())
        self.layer3 = nn.Sequential(nn.Linear(hid_dim2,out_dim)) 
        self.drp1 = nn.Dropout(dp1)
        self.drp2 = nn.Dropout(dp2)
        self.coeff = nn.Parameter(torch.tensor([par]))

    def forward(self,X):
        x = self.layer1(X)
        x = self.drp1(x)
        x = self.layer2(x)
        x = self.drp2(x)
        x = self.layer3(x)
        x = self.coeff*x + (1-self.coeff)*X
        return x


class FullyConnected_dropout3(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim,par,dp1,dp2):
        super(FullyConnected_dropout3,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hid_dim1),nn.BatchNorm1d(hid_dim1),nn.CELU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim1,hid_dim2),nn.BatchNorm1d(hid_dim2),nn.CELU())
        self.layer3 = nn.Sequential(nn.Linear(hid_dim2,out_dim)) 
        self.drp1 = nn.Dropout(dp1)
        self.drp2 = nn.Dropout(dp2)
        self.coeff = nn.Parameter(torch.tensor([par]))

    def forward(self,X):
        x = self.layer1(X)
        x = self.drp1(x)
        x = self.layer2(x)
        x = self.drp2(x)
        x = self.layer3(x)
        x = self.coeff*x + (1-self.coeff)*X
        return x

class FullyConnected_dropout2(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim,par,dp1,dp2):
        super(FullyConnected_dropout2,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hid_dim1),nn.BatchNorm1d(hid_dim1),nn.CELU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim1,hid_dim2),nn.BatchNorm1d(hid_dim2),nn.CELU())
        self.layer3 = nn.Sequential(nn.Linear(hid_dim2,out_dim)) 
        self.drp1 = nn.Dropout(dp1)
        self.drp2 = nn.Dropout(dp2)
        self.coeff = nn.Parameter(torch.tensor([par]))

    def forward(self,X):
        x = self.layer1(X)
        x = self.drp1(x)
        x = self.layer2(x)
        x = self.drp2(x)
        x = self.layer3(x)
        x = self.coeff*x + X
        return x

class FClayer(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim,par,dp1,dp2):
        super(FClayer,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hid_dim1),nn.BatchNorm1d(hid_dim1),nn.PReLU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim1,hid_dim2),nn.BatchNorm1d(hid_dim2),nn.PReLU())
        self.layer3 = nn.Sequential(nn.Linear(hid_dim2,out_dim)) 
        self.drp1 = nn.Dropout(dp1)
        self.drp2 = nn.Dropout(dp2)
        self.coeff = nn.Parameter(torch.tensor([par]))

    def forward(self,X):
        x = self.layer1(X)
        x = self.drp1(x)
        x = self.layer2(x)
        x = self.drp2(x)
        x = self.layer3(x)
        x = self.coeff*x + X
        return x
    #print("Finished")
class FullyConnected2(nn.Module):
    def __init__(self,in_dim,hid_dim1,out_dim,par):
        super(FullyConnected2,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hid_dim1),nn.BatchNorm1d(hid_dim1),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim1,out_dim))
        self.coeff = nn.Parameter(torch.tensor([par]))

    def forward(self,X):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.coeff*x + (1-self.coeff)*X
        return x


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        return loss_contrastive

def cal_dist(sameid,dis,bts):
    dist_loss = 0
    compl = set(np.arange(bts))
    for k in range(bts):
        sid = sameid[k]
        modified_array = sid % bts
        modified_array = np.array(modified_array)
        
        d_similar = torch.sum(dis[k][modified_array])
        
        s1 = set(list(modified_array))
        nsi = list(compl-s1)
        #len_nsi = len(nsi)
        #diff_img_index = np.array(random.sample(range(0,len_nsi),10))
        d_diff = dis[np.array(nsi)]
        dist_loss += torch.sum(torch.clamp(torch.min(d_diff) - d_diff, min=0.0))
        
        dist_loss += d_similar
        #for j in range(dnum):
            #dist_loss += contrast_loss1(differ_img[j].reshape(ddim,1),fc_img1[k].reshape(dim,1),1)
    return dist_loss

#class ConvNet(nn.Module):
