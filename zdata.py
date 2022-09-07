import os

import clip
import torch
import ssl
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import seaborn
seaborn.set()

from utils.read_write_data import read_json, makedir, save_dict, write_txt,read_dict

print(torch.__version__)
print(torch.version.cuda)
torch.ones((4,5)).cuda()
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_folder = colab_root_folder = os.getcwd()

ssl._create_default_https_context = ssl._create_unverified_context
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50x64', device)
print(device)

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

def get_img(image_data_name,captions,num,train=False):
    
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
        torch.save(img_input,"temp2/"+prefix+"_img"+str(i)+".pt")

#get_data(img_name,capt)
def get_text(image_data_name,captions,train=False):
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
        torch.save(text_inputs,"temp2/"+prefix+"_text.pt")
    else:
        for i in range(len(captions)):
            all_text.append(captions[i])
        if prompt != "":
            text_inputs = torch.cat([clip.tokenize(prompt+c,truncate=True) for c in all_text]).to(device)
        else:
            text_inputs = torch.cat([clip.tokenize(c,truncate=True) for c in all_text]).to(device)
        print("text: ",text_inputs.shape)
        torch.save(text_inputs,"temp2/"+prefix+"_text.pt")
get_text(img_name,capt,True)
get_img(img_name,capt,34060,True)
get_img(img_names,capti,3074,False)
get_text(img_names,capti,False)



print("Loading the data...")
for i in range(30,31):
    imseg = torch.load("temp2/test_img"+str(i)+".pt")#group of 100 images
    imseg = imseg.cuda()
    print(imseg.shape)
    to_save = []
    for j in range(4):
        with torch.no_grad():

            img_features = model.encode_image(imseg[j*20:(j+1)*20])
            print(img_features.shape)
            to_save.append(img_features)
    to_save = torch.cat(to_save)
    print(to_save.shape)
    torch.save(to_save,"encode/img2/test_img"+str(i)+".pt")
    print(f"finished: {i+1} / 31")


print("Loading the data...")
alltxt = torch.load("temp2/test_text"+".pt")
alltxt = alltxt.cuda()
print(alltxt.shape)
sz = 62
alldata = []
for i in range(100):
    with torch.no_grad():
        txt_features = model.encode_text(alltxt[i*sz:(i+1)*sz])
        print(txt_features.shape)
        torch.save(txt_features,"encode/text2/test_text"+str(i)+".pt")
        alldata.append(txt_features)
    print(f"finished: {i+1} / 100")

alldata = []
for i in range(100):
    txtseg = torch.load("encode/text2/test_text"+str(i)+".pt")
    txtseg = txtseg.cuda()
    alldata.append(txtseg)
    
alldata = torch.cat(alldata)
print(alldata.shape)
torch.save(alldata,"encode/text2/test_text"+".pt")

alldata = []
for i in range(31):
    imgseg = torch.load("encode/img2/test_img"+str(i)+".pt")
    imgseg = imgseg.cuda()
    print(imgseg.shape)
    alldata.append(imgseg)
    print(f"finished: {i+1} / 31")
alldata = torch.cat(alldata)
print(alldata.shape)
torch.save(alldata,"encode/img2/test_img"+".pt")




















print("Loading the data...")
for i in range(340,341):
    imseg = torch.load("temp2/train_img"+str(i)+".pt")#group of 100 images
    imseg = imseg.cuda()
    print(imseg.shape)
    to_save = []
    for j in range(3):
        with torch.no_grad():

            img_features = model.encode_image(imseg[j*20:(j+1)*20])
            print(img_features.shape)
            to_save.append(img_features)
    to_save = torch.cat(to_save)
    print(to_save.shape)
    torch.save(to_save,"encode/img2/train_img"+str(i)+".pt")
    #with torch.no_grad():

        #img_features = model.encode_image(imseg)
        #print(img_features.shape)
        #torch.save(img_features,"encode/img2/train_img"+str(i)+".pt")
    print(f"finished: {i+1} / 341")


print("Loading the data...")
alltxt = torch.load("temp2/train_text"+".pt")
alltxt = alltxt.cuda()
print(alltxt.shape)
sz = 136
alldata = []
for i in range(135,501):
    with torch.no_grad():
        txt_features = model.encode_text(alltxt[i*sz:(i+1)*sz])
        print(txt_features.shape)
        torch.save(txt_features,"encode/text2/train_text"+str(i)+".pt")
        alldata.append(txt_features)
    print(f"finished: {i+1} / 501")

for i in range(501):
    txtseg = torch.load("encode/text2/train_text"+str(i)+".pt")
    txtseg = txtseg.cuda()
    alldata.append(txtseg)
alldata = torch.cat(alldata)
print(alldata.shape)
torch.save(alldata,"encode/text2/train_text"+".pt")

alldata = []
for i in range(341):
    imgseg = torch.load("encode/img2/train_img"+str(i)+".pt")
    imgseg = imgseg.cuda()
    print(imgseg.shape)
    alldata.append(imgseg)
    print(f"finished: {i+1} / 341")
alldata = torch.cat(alldata)
print(alldata.shape)
torch.save(alldata,"encode/img2/train_img"+".pt")

