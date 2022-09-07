import os
import clip
import torch
import ssl
from PIL import Image
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from matplotlib import pyplot as plt
import seaborn
seaborn.set()

from tqdm.notebook import trange, tqdm

import torchvision.transforms as transforms
import clip_model
from utils.read_write_data import read_json, makedir, save_dict, write_txt,read_dict
import cal
import random

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_folder = colab_root_folder = os.getcwd()

ssl._create_default_https_context = ssl._create_unverified_context
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)

test_path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES/processed_data"
test_data_name = read_dict(test_path+"/train_save.pkl")
#print(type(test_data_name))
print(test_data_name.keys())
print(test_data_name['id'][:20])
print(test_data_name['img_path'][:20])
print(test_data_name['same_id_index'][:10])
#print(test_data_name['lstm_caption_id'][:10])
#print(test_data_name['caption_label'][:20])

print(test_data_name['captions'][:10])
def selflab(list):
    classes = 0
    labself = []
    mark = list[0]
    labself.append(classes)
    for i in range(1,len(list)):
        if mark != list[i]:
            mark = list[i]
            classes += 1
        labself.append(classes)

label = test_data_name['id']
same_id = test_data_name['same_id_index']
img_name = test_data_name['img_path']
print(img_name[len(img_name)-10:len(img_name)])
capt = test_data_name['captions']
print(len(same_id),len(img_name),len(capt))
#test_data_name = readname(test_path)
#print(test_data_name[0])
test_info = read_dict(test_path+"/test_save.pkl")
print(type(test_info))
print(test_info.keys())

print(test_info['id'][:20])
print(test_info['img_path'][:10])

print(test_info['img_caption_index'][:10])
print(test_info['caption_matching_img_index'][:10])
print(test_info['caption_label'][:20])

print(test_info['captions'][:10])
txt_labels, img_labels = test_info['caption_label'],test_info['id']
img_cap_index = test_info['img_caption_index']
cap_match_imgind = test_info['caption_matching_img_index']
img_names = test_info['img_path']
capti = test_info['captions']
txt_labels = np.array(txt_labels)
img_labels = np.array(img_labels)
#print(cap_match_imgind)
print(len(img_names),len(capti))

def get_img(image_data_name,captions,train=False):
    
    path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES/"
    prefix = "test" if train == False else "train"
    #print(len(image_data_name))
    img_path = []
    for i in range(34060):
        img_path.append(image_data_name[i*2])
    #print(img_path[len(img_path)-6:len(img_path)])
    #print(img_name)
    lis = [i*100 for i in range(34000//100+1)]
    lis += [34060]
    #print(lis)
    
    for i in range(len(lis)-1):
        img_seg = img_path[lis[i]:lis[i+1]]
        all_images = []
        for img in img_seg:
            image = Image.open(path+img)
            image_input = preprocess(image).unsqueeze(0).to(device)
            #print(image_input.shape)
            all_images.append(image_input)
        img_input = torch.cat(all_images)
        print("img: ",img_input.shape)
        torch.save(img_input,"temp/"+prefix+"_img"+str(i)+".pt")

#get_data(img_name,capt)
def get_text(image_data_name,captions,train=False):
    all_text = []
    prefix = "test" if train == False else "train"
    prompt = ""
    lis = [0,8000,16000,24000,32000,40000,48000,56000,62000,68120]
    for i in range(len(lis)-1):
        caption_seg = captions[lis[i]:lis[i+1]]
        all_text.append(caption_seg)
    text_inputs = None
    if prompt != "":
        text_inputs = torch.cat([clip.tokenize(prompt+c,truncate=True) for c in all_text]).to(device)
    else:
        text_inputs = torch.cat([clip.tokenize(c,truncate=True) for c in all_text]).to(device)
    print("text: ",text_inputs.shape)
    torch.save(text_inputs,"temp/"+prefix+"_text.pt")
get_text(img_name,capt,True)
get_img(img_name,capt,True)

print("Loading the data...")
for i in range(341):
    imseg = torch.load("temp/train_img"+str(i)+".pt")#group of 100 images
    with torch.no_grad():

        img_features = model.encode_image(imseg)
        print(img_features.shape)
        torch.save(img_features,"encode/img/train_img"+str(i)+".pt")
    print(f"finished: {i+1} / 341")


def load_data():
    img_all = []

    for i in range(341):
        
        imseg = torch.load("encode/img/train_img"+str(i)+".pt")
        #print(type(txseg))
        img_all.append(imseg)
        
    encoded_img = torch.cat(img_all)

    text_inputs = torch.load("temp/train_text.pt")
    lis = [i*1000 for i in range(68)]
    lis += [68120]
    #print(lis)
    '''
    for i in range(len(lis)-1):
        text_seg = text_inputs[lis[i]:lis[i+1]]
        with torch.no_grad():
            text_features = model.encode_text(text_seg)
            torch.save(text_features,"encode/text/train_text"+str(i)+".pt")
        print(f"Finished: {i+1} / {len(lis)-1}")
    '''
    text_all = []
    for i in range(len(lis)-1):
        
        txseg = torch.load("encode/text/train_text"+str(i)+".pt")
        #print(type(txseg))
        text_all.append(txseg)
    encoded_text = torch.cat(text_all)
    ig_all = []
    for i in range(34060):
        #print(encoded_img[i].reshape((1,768)).shape)
        ig_all.append(encoded_img[i].reshape((1,768)))
        ig_all.append(encoded_img[i].reshape((1,768)))
        #break
    encoded_img = torch.cat(ig_all)
    print(encoded_img.shape,encoded_text.shape)

    print("Successfully loaded!")
    return encoded_img,encoded_text


def getbatch_data(img,text,batch_size,start):
    ig,tx = img.shape[0],text.shape[0]
    return img[start:min(start+batch_size,ig)],text[start:min(start+batch_size,tx)]
def get_label(lab,batch_size,start):
    lb = lab.shape[0]
    return lab[start:min(start+batch_size,lb)]

'''shuffle the training data'''
'''
state = np.random.get_state()
np.random.shuffle(encoded_img)
np.random.set_state(state)
np.random.shuffle(encoded_text)
np.random.set_state(state)
np.random.shuffle(label)
print(len(label))
print(label[:20])
'''
'''use the test data to eval
using the encoded test img/text, feeding into the trained fc layer'''

'''
class FullyConnected(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim):
        super(FullyConnected,self).__init__()
        self.layer1 = nn.Linear(in_dim,hid_dim1)
        self.layer2 = nn.Linear(hid_dim1,hid_dim2)
        self.layer3 = nn.Linear(hid_dim2,out_dim)
    
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
'''
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

class FullyConnected(nn.Module):
    def __init__(self,in_dim,hid_dim1,hid_dim2,out_dim,par):
        super(FullyConnected,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hid_dim1),nn.BatchNorm1d(hid_dim1),nn.PReLU())
        self.layer2 = nn.Sequential(nn.Linear(hid_dim1,hid_dim2),nn.BatchNorm1d(hid_dim2),nn.PReLU())
        self.layer3 = nn.Sequential(nn.Linear(hid_dim2,out_dim)) 
        self.drp1 = nn.Dropout(0.1)
        self.drp2 = nn.Dropout(0.1)
        self.coeff = nn.Parameter(torch.tensor([par]))

    def forward(self,X):
        x = self.layer1(X)
        x = self.drp1(x)
        x = self.layer2(x)
        x = self.drp2(x)
        x = self.layer3(x)
        x = self.coeff*x + (1-self.coeff)*X
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
'''
load the test data
'''

img_all = []
for i in range(10):
    
    imseg = torch.load("Encoded_test_img"+str(i)+".pt")
    #print(type(txseg))
    img_all.append(imseg)
text_all = []
for i in range(4):
    
    txseg = torch.load("Encoded_test_text"+str(i)+".pt")
    print(type(txseg))
    text_all.append(txseg)
image_features = torch.cat(img_all)
text_features = torch.cat(text_all)
print(image_features.shape,text_features.shape)
# Custom Contrastive Loss
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


def train(e_num,a,b,lr1,lr2,batch_size,prt,eimg,etext,image_features,text_features):
    num_epochs = e_num
    epoch_iterator = trange(num_epochs)
    loss_fn = nn.CrossEntropyLoss()
    num_epo = 0
    #optimizer = optim.Adam(linear_model3.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-9)
    #optimizer = optim.SGD(linear_model3.parameters(), lr=0.005, momentum=0.9)
    optimizer = optim.Adam([
        {'params':linear_model3.parameters(), 'lr':lr1, 'betas':(0.9,0.95),'weight_decay':1e-9},
        {'params':linear_model2.parameters(), 'lr':lr2, 'betas':(0.9,0.95),'weight_decay':1e-9}
    ])
    '''
    optimizer = optim.SGD([
        {'params':linear_model3.parameters(), 'lr':lr1, 'momentum':0.9},
        {'params':linear_model2.parameters(), 'lr':lr2, 'momentum':0.9}
        ])
    '''
    all_index = [i for i in range(eimg.shape[0])]
    test_path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES/processed_data"
    test_data_name = read_dict(test_path+"/train_save.pkl")
    #contrast_loss1 = ContrastiveLoss()
    #contrast_loss2 = ContrastiveLoss()
    label = test_data_name['id']
    same_id = test_data_name['same_id_index']
    same_id = np.array(same_id)
    best_test_r1 = 0
    best_test_map = 0

    tr_num = eimg.shape[0]
    for epoch in epoch_iterator:
        # Train
        #encoded_img,encoded_text = load_data()
        '''
        state = np.random.get_state()
        np.random.shuffle(eimg)
        np.random.set_state(state)
        np.random.shuffle(etext)
        np.random.set_state(state)
        np.random.shuffle(label)
        np.random.set_state(state)
        np.random.shuffle(same_id)
        '''
        #txind = np.array([2*i+num_epo%2 for i in range(tr_num)])
        #enc_text = etext[txind]
        #labels = np.arange(batch_size)
        for i in range(train_data_num//batch_size):
            linear_model3.train()
            linear_model2.train()
            image_feature,text_feature = getbatch_data(eimg,etext,batch_size,i*batch_size)
            #labels = get_label(label,batch_size,i*batch_size)
            labels = np.arange(image_feature.shape[0])
            
            sam = get_label(same_id,batch_size,i*batch_size)
            #labels = get_label(cap_match_imgind,batch_size,i*batch_size)
            # normalized features
            c_loss = 0
            fc_img1 = linear_model3(image_feature)
            fc_text1 = linear_model2(text_feature)
            
            bts = image_feature.shape[0]
            #lck_img_index = np.array(random.sample(range(0,bts),bts//2))
            sq_img = fc_img1@fc_img1.t()
            sq_tx = fc_text1@fc_text1.t()

            dist = torch.diagonal(sq_img) - 2.0*(fc_text1@fc_img1.t()) + torch.unsqueeze(torch.diagonal(sq_tx),1)
            dis_img = torch.diagonal(sq_img) - 2.0*sq_img + torch.unsqueeze(torch.diagonal(sq_img),1)
            dis_text = torch.diagonal(sq_tx) - 2.0*sq_tx + torch.unsqueeze(torch.diagonal(sq_tx),1)
            
            sdig = torch.diag(dist)
            ndig = dist-sdig
            #c_loss += torch.sum(torch.clamp(torch.min(ndig) - ndig, min=0.0))
            d_pair = torch.sum(sdig/sdig.norm())
            #c_loss += d_pair
            
            dis_img = dis_img / dis_img.norm()
            dis_text = dis_text / dis_text.norm()

            #c_loss += cal_dist(sam,dis_text,bts)
            #c_loss += cal_dist(sam,dis_img,bts)

            fc_img1 = fc_img1 / fc_img1.norm(dim=1, keepdim=True)
            fc_text1 = fc_text1 / fc_text1.norm(dim=1, keepdim=True)
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            logit_scales = logit_scale.exp()
            logits_per_image1 = fc_img1 @ fc_text1.t()
            logits_per_image1 *= logit_scales
            logits_per_text1 = logits_per_image1.t()

            loss_img = loss_fn(logits_per_image1.to(torch_device),torch.tensor(labels).to(torch_device).long())
            loss_text = loss_fn(logits_per_text1.to(torch_device),torch.tensor(labels).to(torch_device).long())
            loss = (loss_img + loss_text)/2
            loss = loss + c_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % prt == 0:
                with torch.no_grad():
                    param1 = list(linear_model3.named_parameters())
                    print("alpha: ",param1[0][1].item())
                    param2 = list(linear_model2.named_parameters())
                    print("beta: ",param2[0][1].item())

                    fc_img2 = linear_model3(image_features)
                    fc_text2 = linear_model2(text_features)
                    fc_img2 = fc_img2/fc_img2.norm(dim=1, keepdim=True)
                    fc_text2 = fc_text2/fc_text2.norm(dim=1, keepdim=True)
                    logits_per_image2 = logit_scales * fc_img2 @ fc_text2.t()
                    logits_per_text2 = logits_per_image2.t()
                    probs2 = logits_per_text2.softmax(dim=-1).cpu().detach().numpy()
                    t2i_cmc, t2i_map = cal.evaluate(probs2, txt_labels, img_labels)
                    strr = "t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, map: {:.4}".format(t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_map)
                    print(strr)
                    print(f"Training: loss:{loss.item()}, {i*batch_size}/{train_data_num}, epoch: {num_epo}")
                    if best_test_r1 < t2i_cmc[0]:
                        best_test_r1 = t2i_cmc[0]
                        #print("############################")
                        print("\nBest @Ri: ",best_test_r1)
                        print("\n")
                        torch.save(linear_model3.state_dict(),'linear_train_img_r1.pt')
                        torch.save(linear_model2.state_dict(),'linear_train_text_r1.pt')
                    
        num_epo += 1
        
        '''
        
        data_iterator = tqdm(trainloader)

        for x, y in data_iterator:
            total_steps += 1
            x, y = x.to(torch_device), y.to(torch_device)
            logits = model(x)
            loss = torch.mean(F.cross_entropy(logits, y))
            accuracy = torch.mean((torch.argmax(logits, dim=-1) == y).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_iterator.set_postfix(loss=loss.item(), train_acc=accuracy.item())

            if total_steps % train_logfreq == 0:
                losses.append(loss.item())
                train_acc.append(accuracy.item())

        # Validation
        val_acc = []
        model.eval()
        for x, y in testloader:
            x, y = x.to(torch_device), y.to(torch_device)
            with torch.no_grad():
            logits = model(x)
            accuracy = torch.mean((torch.argmax(logits, dim=-1) == y).float())
            val_acc.append(accuracy.item())
        model.train()

        all_val_acc.append(np.mean(val_acc))
        # Save best model
        if np.mean(val_acc) > best_val_acc:
            best_val_acc = np.mean(val_acc)
        '''
        #break
        #epoch_iterator.set_postfix(val_acc=np.mean(val_acc), best_val_acc=best_val_acc)
    

total_steps = 0

train_logfreq = 100
losses = []
train_acc = []
all_val_acc = []
best_val_acc = 0

train_data_num = 68120
batch_size = 1703

test_data_num = 3074 #you should use real validation data

encoded_img,encoded_text = load_data()
    #torch.save(imseg,"encode/img/train_img"+str(i)+".pt")
linear_model1 = FullyConnected(768,960,1024,1296,0.1)
linear_model2 = FullyConnected(768,3072,2048,768,0.64)
linear_model3 = FullyConnected(768,3072,2048,768,0.76)

linear_model2 = FullyConnected2(768,2048,768,0.55)
linear_model3 = FullyConnected2(768,2048,768,0.78)

linear_model2 = FullyConnectedApp(768,2048,1024,232)
linear_model3 = FullyConnectedApp(768,2048,1024,232)
train(70,0.4,0.2,2e-5,2e-5,batch_size,5,encoded_img,encoded_text,image_features,text_features)

plt.plot(losses)
plt.title('Train Loss')
plt.figure()
#plt.plot(train_acc)
#plt.title('Train Accuracy')
#plt.figure()
plt.plot(all_val_acc)
plt.title('Val Accuracy')


#ind2w = read_dict(test_path+"/ind2word.pkl")
#print(ind2w[:200])