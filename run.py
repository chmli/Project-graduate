from gettext import gettext
from importlib.resources import read_text
import os
from tkinter import image_names
from tokenize import group
from unittest import TestProgram

from matplotlib.pyplot import text
import clip
import torch
import ssl
from PIL import Image
import json

import einops
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

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_folder = colab_root_folder = os.getcwd()

ssl._create_default_https_context = ssl._create_unverified_context
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)



def readname(path):
    return os.listdir(path)
train_path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES" 
train_data_name = readname(train_path)
train_data_name = sorted(train_data_name[1:],reverse=False)
print(len(train_data_name))

print(train_data_name[:15])

test_path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES/processed_data"

test_data_name = read_dict(test_path+"/test_save.pkl")
print(type(test_data_name))
print(test_data_name.keys())

print(test_data_name['id'][:20])
print(test_data_name['img_path'][:10])

print(test_data_name['img_caption_index'][:10])
print(test_data_name['caption_matching_img_index'][:10])
print(test_data_name['caption_label'][:20])

print(test_data_name['captions'][:10])
txt_labels, img_labels = test_data_name['caption_label'],test_data_name['id']
img_cap_index = test_data_name['img_caption_index']
cap_match_imgind = test_data_name['caption_matching_img_index']
img_name = test_data_name['img_path']
capt = test_data_name['captions']
print(len(img_name),len(capt))
#test_data_name = readname(test_path)
#print(test_data_name[0])
def get_img(image_data_name,captions,train=False):
    
    path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES/"
    prefix = "test" if train == False else "train"
    #print(len(image_data_name))
    lis = [0,300,600,900,1200,1500,1800,2100,2400,2700,3074]
    
    for i in range(len(lis)-1):
        img_seg = image_data_name[lis[i]:lis[i+1]]
        all_images = []
        for img in img_seg:
            image = Image.open(path+img)
            image_input = preprocess(image).unsqueeze(0).to(device)
            #print(image_input.shape)
            all_images.append(image_input)
        img_input = torch.cat(all_images)
        print("img: ",img_input.shape)
        torch.save(img_input,prefix+"_img"+str(i)+".pt")
    '''
    for img in image_data_name:
        
        image = Image.open(path+img)
        image_input = preprocess(image).unsqueeze(0).to(device)
        #print(image_input.shape)
        all_images.append(image_input)
    
    img_input = torch.cat(all_images)
    
    print("img: ",img_input.shape)
    torch.save(img_input,prefix+"_img.pt")
    '''

#get_data(img_name,capt)
def get_text(image_data_name,captions,train=False):
    all_text = []
    prefix = "test" if train == False else "train"
    prompt = ""
    for cap in captions:
        all_text.append(prompt+cap)

    text_inputs = torch.cat([clip.tokenize(c,truncate=True) for c in all_text]).to(device)
    print("text: ",text_inputs.shape)
    torch.save(text_inputs,prefix+"_text.pt")
get_text(img_name,capt)
get_img(img_name,capt)

print("Loading the data...")
#lis = [0,300,600,900,1200,1500,1800,2100,2400,2700,3074]
#img_all = []
#for i in range(len(lis)-1):
    
    #imseg = torch.load("test_img"+str(i)+".pt")
    #print(type(txseg))
    #img_all.append(imseg)
#img_input = torch.cat(img_all)
img_input = torch.load("test_img.pt")
text_inputs = torch.load("test_text.pt")
print("Successfully loaded!")


imgin = img_input[:]
textin = text_inputs[:]
print(imgin.shape,textin.shape)

lo_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



#lis = [0,75,150,225,300,375,450,525,600,675,750,825,900,975,1050,1125,1200,1275,1350,1425,1500,1575,1650,1725,1800,1875,1950,2025,2100,2175,2250,2400,2475,2550,2625,2700,2775,2850,2925,3000,3074]
lis = [0,300,600,900,1200,1500,1800,2100,2400,2700,3074]

for i in range(len(lis)-1):
    img_seg = imgin[lis[i]:lis[i+1]]
    with torch.no_grad():

        img_features = model.encode_image(img_seg)
        torch.save(img_features,"Encoded_test_img"+str(i)+".pt")
    print("Finished")
img_all = []
for i in range(len(lis)-1):
    
    imseg = torch.load("Encoded_test_img"+str(i)+".pt")
    #print(type(txseg))
    img_all.append(imseg)


lis = [0,1500,3000,4500,6156]
for i in range(len(lis)-1):
    text_seg = textin[lis[i]:lis[i+1]]
    with torch.no_grad():

        text_features = model.encode_text(text_seg)
        torch.save(text_features,"Encoded_test_text"+str(i)+".pt")
    print("Finished")

text_all = []
for i in range(len(lis)-1):
    
    txseg = torch.load("Encoded_test_text"+str(i)+".pt")
    print(type(txseg))
    text_all.append(txseg)

    #print("Finished")

image_features = torch.cat(img_all)
text_features = torch.cat(text_all)
print(image_features.shape,text_features.shape)
print(img_features[0][:10],text_features[0][:10])
# normalized features
image_features = image_features / image_features.norm(dim=1, keepdim=True)
text_features = text_features / text_features.norm(dim=1, keepdim=True)

# cosine similarity as logits
logit_scale = lo_scale.exp()

logits_per_image = lo_scale * image_features @ text_features.t()
logits_per_text = logits_per_image.t()

#logits_per_image, logits_per_text = model(imgin, textin)
probs2 = logits_per_text.softmax(dim=-1).cpu().detach().numpy()
#print(probs2)
#top = torch.tensor(probs2[:]).topk(tp)

txt_labels = np.array(txt_labels)
img_labels = np.array(img_labels)
print(txt_labels.shape,img_labels.shape,probs2.shape)
num_imgs = 100000
numtextin = 200000
t2i_cmc, t2i_map = cal.evaluate(probs2[:,:num_imgs], txt_labels[:numtextin], img_labels[:num_imgs])
strr = "t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, map: {:.4}".format(t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_map)
print(strr)

#ind2w = read_dict(test_path+"/ind2word.pkl")
#print(ind2w[:200])