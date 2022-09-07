import os
from turtle import forward

import clip
import torch
import ssl
from PIL import Image
import PIL
import numpy as np
from matplotlib import pyplot as plt
import seaborn
seaborn.set()

from utils.read_write_data import read_json, makedir, save_dict, write_txt,read_dict
import torchvision.transforms as T

modelname = 'ViT-L/14'


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_folder = colab_root_folder = os.getcwd()

ssl._create_default_https_context = ssl._create_unverified_context
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(modelname, device)
print(device)




test_path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES/processed_data"
test_data_name = read_dict(test_path+"/train_save.pkl")

label = test_data_name['id']
same_id = test_data_name['same_id_index']
img_name = test_data_name['img_path']

capt = test_data_name['captions']


test_info = read_dict(test_path+"/test_save.pkl")

txt_labels, img_labels = test_info['caption_label'],test_info['id']
img_cap_index = test_info['img_caption_index']
cap_match_imgind = test_info['caption_matching_img_index']
img_names = test_info['img_path']
capti = test_info['captions']
txt_labels = np.array(txt_labels)
img_labels = np.array(img_labels)
#print(cap_match_imgind)
print(len(img_names),len(capti),len(cap_match_imgind))

def get_img(image_data_name,captions,num,dir,train=False):
    
    path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES/"
    prefix = "test" if train == False else "train"
    #print(len(image_data_name))
    img_path = []
    if prefix == "train":
        for i in range(num):
            img_path.append(image_data_name[i*2])
    else:
        for i in range(num):
            img_path.append(image_data_name[i])
    
    #print(img_path[len(img_path)-6:len(img_path)])
    #print(img_name)
    lis = [i*100 for i in range(num//100+1)]
    lis += [num]
    #print(lis)
    hflip = T.RandomHorizontalFlip(p=0.5)
    vflip = T.RandomVerticalFlip(p=0.5)
    rotate = T.RandomRotation(degrees=(-80,80),expand=False)
    sharp = T.RandomAdjustSharpness(sharpness_factor=2,p=1)
    contrast = T.RandomAutocontrast(p=1)
    equal = T.RandomEqualize(p=1)
    ct = 1
    for i in range(len(lis)-1):
        img_seg = img_path[lis[i]:lis[i+1]]
        all_images = []
        alligr,alligh,alligv,alligs,alligc,allige = [],[],[],[],[],[]
        for img in img_seg:
            image = Image.open(path+img)

            #image_input = preprocess(image).unsqueeze(0).to(device)
            #print(image_input.shape)
            #rimg = preprocess(rotate(image)).unsqueeze(0).to(device)
            #himg = preprocess(hflip(image)).unsqueeze(0).to(device)
            #vimg = preprocess(vflip(image)).unsqueeze(0).to(device)
            #simg = preprocess(sharp(image)).unsqueeze(0).to(device)
            cimg = preprocess(contrast(image)).unsqueeze(0).to(device)
            #eqimg = preprocess(equal(image)).unsqueeze(0).to(device)
            #all_images.append(image_input)
            
            #alligr.append(rimg)
            #alligh.append(himg)
            #alligv.append(vimg)
            #alligs.append(simg)
            alligc.append(cimg)
            #allige.append(eqimg)
        print(ct)
        ct += 1
        #img_input = torch.cat(all_images)
        #igin_r = torch.cat(alligr)
        #igin_h = torch.cat(alligh)
        #igin_v = torch.cat(alligv)
        #igin_s = torch.cat(alligs)
        igin_c = torch.cat(alligc)
        #igin_e = torch.cat(allige)
        #print("img: ",img_input.shape)

        #torch.save(img_input,dir+prefix+"_img"+str(i)+".pt")
        #torch.save(igin_r,dir+prefix+"_imgr"+str(i)+".pt")
        #torch.save(igin_h,dir+prefix+"_imgh"+str(i)+".pt")
        #torch.save(igin_v,dir+prefix+"_imgv"+str(i)+".pt")
        #torch.save(igin_s,dir+prefix+"_imgs"+str(i)+".pt")
        torch.save(igin_c,dir+prefix+"_imgc"+str(i)+".pt")
        #torch.save(igin_e,dir+prefix+"_imge"+str(i)+".pt")
import os

allf = os.listdir("D:/MACHINE LEARNING/Proj/start/encode/img/")
import torch
for i in range(len(allf)):
    #print(allf[i][:5])
    if allf[i][:5] == 'train':
        #print()
        if allf[i][9] == 'c':
            print(allf[i])
            
            src = "D:/MACHINE LEARNING/Proj/start/encode/img/"+allf[i]
            
            os.rename(src,src.replace('c','e'))
            #print(allf[i])
#get_data(img_name,capt)
def get_text(image_data_name,captions,dir,train=False):
    all_text = []
    prefix = "test" if train == False else "train"
    prompt = ""
    if prefix == "train":
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
        torch.save(text_inputs,dir+prefix+"_text.pt")
    else:
        for i in range(len(captions)):
            all_text.append(captions[i])
        if prompt != "":
            text_inputs = torch.cat([clip.tokenize(prompt+c,truncate=True) for c in all_text]).to(device)
        else:
            text_inputs = torch.cat([clip.tokenize(c,truncate=True) for c in all_text]).to(device)
        print("text: ",text_inputs.shape)
        torch.save(text_inputs,dir+prefix+"_text.pt")

direc = "temp/"
get_text(img_name,capt,direc,True)
get_img(img_name,capt,34060,direc,True)
get_img(img_names,capti,3074,direc,False)
get_text(img_names,capti,direc,False)


seg = 20
dimg = "img/"
dtext = "text/"
print("Loading the data...")
def modelforward(direc,st):
    for i in range(30):
        imseg = torch.load(direc+"test_img"+st+str(i)+".pt")#group of 100 images
        imseg = imseg.cuda()
        print(imseg.shape)
        to_save = []
        for j in range(5):
            with torch.no_grad():

                img_features = model.encode_image(imseg[j*seg:(j+1)*seg])
                print(img_features.shape)
                to_save.append(img_features)
        to_save = torch.cat(to_save)
        print(to_save.shape)
        torch.save(to_save,"encode/"+dimg+"test_img"+st+str(i)+".pt")
        print(f"finished: {i+1} / 31")


modelforward(direc,"")

print("Loading the data...")
alltxt = torch.load(direc+"test_text"+".pt")
alltxt = alltxt.cuda()
print(alltxt.shape)

sz = 76
alldata = []
for i in range(81):
    with torch.no_grad():
        txt_features = model.encode_text(alltxt[i*sz:(i+1)*sz])
        print(txt_features.shape)
        torch.save(txt_features,"encode/"+dtext+"test_text"+str(i)+".pt")
        alldata.append(txt_features)
    print(f"finished: {i+1} / 81")

alldata = []
for i in range(81):
    txtseg = torch.load("encode/"+dtext+"test_text"+str(i)+".pt")
    txtseg = txtseg.cuda()
    alldata.append(txtseg)
    
alldata = torch.cat(alldata)
print(alldata.shape)
torch.save(alldata,"encode/"+dtext+"test_text"+".pt")

def storet(dimg,st):
    alldata = []
    for i in range(31):
        imgseg = torch.load("encode/"+dimg+"test_img"+st+str(i)+".pt")
        imgseg = imgseg.cuda()
        #print(imgseg.shape)
        alldata.append(imgseg)
        #print(f"finished: {i+1} / 31")
    alldata = torch.cat(alldata)
    print(alldata.shape)
    torch.save(alldata,"encode/"+dimg+"test_img"+st+".pt")
storet(dimg,"r")
storet(dimg,"h")
storet(dimg,"v")
storet(dimg,"")



















szz = 20
print("Loading the data...")
def trmodelfor(direc,st):
    for i in range(340,341):#43434403043043043043043030430430430430430403403
        
        imseg = torch.load(direc+"train_img"+st+str(i)+".pt")#group of 100 images
        imseg = imseg.cuda()
        print(imseg.shape)
        to_save = []
        for j in range(3):
            with torch.no_grad():

                img_features = model.encode_image(imseg[j*szz:(j+1)*szz])
                print(img_features.shape)
                to_save.append(img_features)
        to_save = torch.cat(to_save)
        print(to_save.shape)
        torch.save(to_save,"encode/"+dimg+"train_img"+st+str(i)+".pt")
        #with torch.no_grad():

            #img_features = model.encode_image(imseg)
            #print(img_features.shape)
            #torch.save(img_features,"encode/img2/train_img"+str(i)+".pt")
        print(f"finished: {i+1} / 341")
trmodelfor(direc,"r")
trmodelfor(direc,"h")
trmodelfor(direc,"v")
trmodelfor(direc,"s")
trmodelfor(direc,"c")
trmodelfor(direc,"e")
trmodelfor(direc,"")

print("Loading the data...")
alltxt = torch.load(direc+"train_text"+".pt")
alltxt = alltxt.cuda()
print(alltxt.shape)
sz = 136
alldata = []
for i in range(501):
    with torch.no_grad():
        txt_features = model.encode_text(alltxt[i*sz:(i+1)*sz])
        print(txt_features.shape)
        torch.save(txt_features,"encode/"+dtext+"train_text"+str(i)+".pt")
        alldata.append(txt_features)
    print(f"finished: {i+1} / 501")

alldata = []
for i in range(501):
    txtseg = torch.load("encode/"+dtext+"train_text"+str(i)+".pt")
    txtseg = txtseg.cuda()
    alldata.append(txtseg)
alldata = torch.cat(alldata)
print(alldata.shape)
torch.save(alldata,"encode/"+dtext+"train_text"+".pt")

def stor(dimg,st):
    alldata = []
    for i in range(341):
        imgseg = torch.load("encode/"+dimg+"train_img"+st+str(i)+".pt")
        imgseg = imgseg.cuda()
        #print(imgseg.shape)
        alldata.append(imgseg)
    print(f"finished: ")
    alldata = torch.cat(alldata)
    print(alldata.shape)
    torch.save(alldata,"encode/"+dimg+"train_img"+st+".pt")
stor(dimg,"r")
stor(dimg,"h")
stor(dimg,"v")
stor(dimg,"s")
stor(dimg,"c")
stor(dimg,"e")
stor(dimg,"")


