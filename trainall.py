from audioop import lin2adpcm
from logging.config import valid_ident
import os
import clip
import torch
import ssl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import trange, tqdm
import numpy as np
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
from utils.read_write_data import read_json, makedir, save_dict, write_txt,read_dict
import cal
import finetune as ft
import importlib
import itertools
from PIL import Image
importlib.reload(ft)
ft.pp()
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
#torch_device = 'cpu'
#device = 'cpu'
root_folder = colab_root_folder = os.getcwd()
print(device)

test_path = "D:/MACHINE LEARNING/Proj/start/dataset/CUHK-PEDES/processed_data"
test_data_name = read_dict(test_path+"/train_save.pkl")
tr_img_path = test_data_name['img_path']
'''
print(test_data_name.keys())
print(test_data_name['id'][:20])
print(test_data_name['img_path'][:20])
print(test_data_name['same_id_index'][:10])
print(test_data_name['lstm_caption_id'][:10])
print(test_data_name['caption_label'][:20])
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
'''

label = test_data_name['id']
same_id = test_data_name['same_id_index']
'''
def getpartindex(sid,bts):
    ind = []
    for i in range(len(sid)//bts+1):
        bt = sid[i*bts:min((i+1)*bts,len(sid))]
        m = -1
        for j in range(len(bt)):
            b = bt[j]
            if m < max(b):
                m = max(b)
        ind.append(m)
    print(ind)
    return ind
print(same_id[:25])
print(len(same_id[:25]))
dd = getpartindex(same_id[:25],12)
print(dd)
pdis = getpartindex(same_id,1024)
'''
img_name = test_data_name['img_path']
capt = test_data_name['captions']


#print(img_name[len(img_name)-10:len(img_name)])
#print(len(same_id),len(img_name),len(capt))
#test_data_name = readname(test_path)
#print(test_data_name[0])
test_info = read_dict(test_path+"/test_save.pkl")
'''
print(type(test_info))
print(test_info.keys())
print(test_info['id'][:20])
print(test_info['img_path'][:10])
print(test_info['img_caption_index'][:10])
print(test_info['caption_matching_img_index'][:10])
print(test_info['caption_label'][:20])
print(test_info['captions'][:10])
'''
txt_labels, img_labels = test_info['caption_label'],test_info['id']
img_cap_index = test_info['img_caption_index']
cap_match_imgind = test_info['caption_matching_img_index']
img_names = test_info['img_path']
capti = test_info['captions']
txt_labels = np.array(txt_labels)
img_labels = np.array(img_labels)
#print(cap_match_imgind)
print(len(img_names),len(capti))


def load_data(dim,diri,dirt,st=""):
    encoded_img = torch.load("encode/"+diri+"train_img"+st+".pt")
    encoded_text = torch.load("encode/"+dirt+"train_text"+st+".pt")
    ig_all = []
    for i in range(34060):
        #print(encoded_img[i].reshape((1,768)).shape)
        ig_all.append(encoded_img[i].reshape((1,dim)))
        ig_all.append(encoded_img[i].reshape((1,dim)))
        #break
    encoded_img = torch.cat(ig_all)
    print(encoded_img.shape,encoded_text.shape)
    print("Successfully loaded!")
    if device == 'cpu':
        return encoded_img.cuda().data.cpu(),encoded_text.cuda().data.cpu()
    return encoded_img,encoded_text

def load_img(dim,diri,dirt,st=""):
    encoded_img = torch.load("encode/"+diri+"train_img"+st+".pt")
    ig_all = []
    for i in range(34060):
        #print(encoded_img[i].reshape((1,768)).shape)
        ig_all.append(encoded_img[i].reshape((1,dim)))
        ig_all.append(encoded_img[i].reshape((1,dim)))
        #break
    encoded_img = torch.cat(ig_all)
    #print(encoded_img.shape)
    #print("Successfully loaded!")
    if device == 'cpu':
        return encoded_img.cuda().data.cpu()
    return encoded_img

def getbatch_data(img,text,batch_size,start,dev):
    ig,tx = img.shape[0],text.shape[0]
    rg = 15
    if dev == "cpu":
        return img[start:min(start+rg+batch_size,ig)].cpu(),text[start:min(start+rg+batch_size,tx)].cpu()
    if dev == "cuda":
        return img[start:min(start+rg+batch_size,ig)].cuda(),text[start:min(start+rg+batch_size,tx)].cuda()
def get_label(lab,batch_size,start):
    lb = lab.shape[0]
    return lab[start:min(start+batch_size,lb)]
def getdata(img,text,batch_size,start,dev):
    ig,tx = img.shape[0],text.shape[0]
    if dev == "cpu":
        return img[start:min(start+batch_size,ig)].cpu(),text[start:min(start+batch_size,tx)].cpu()
    if dev == "cuda":
        return img[start:min(start+batch_size,ig)].cuda(),text[start:min(start+batch_size,tx)].cuda()
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
    if device == "cuda":
        return dist_loss.cuda()
    return dist_loss

def triplet_mask(labels):
    labels = torch.tensor(labels)
    indices_equal = torch.eye(labels.shape[0])
    indices_not_equal =torch.logical_not(indices_equal)
	# 因为最后得到一个3D的mask矩阵(i, j, k)，增加一个维度，则 i_not_equal_j 在第三个维度增加一个即，(batch_size, batch_size, 1), 其他同理
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2) 
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)
	# 想得到i!=j!=k, 三个不等取and即可, 最后可以得到当下标（i, j, k）不相等时才取True
    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
	# 同样根据labels得到对应i=j, i!=k
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))
	# mask即为满足上面两个约束，所以两个3D取and
    mask = torch.logical_and(distinct_indices, valid_labels)
    #print(mask[mask!=0][:10])
    return mask.to(device)
def pairwise_distance(e1,e2):
    dotp = e1@e2.t()
    '''dist = torch.diagonal(sq_img) - 2.0*(fc_text1@fc_img1.t()) + torch.unsqueeze(torch.diagonal(sq_tx),1)'''
    return torch.diagonal(e1@e1.t()) - 2.0*(dotp) + torch.unsqueeze(torch.diagonal(e2@e2.t()),1)
def batch_all_triplet_loss(labels, embed1,embed2, margin, squared=False):
    '''
        triplet loss of a batch
        -------------------------------
        Args:
            labels:     标签数据，shape = （batch_size,）
            embeddings: 提取的特征向量， shape = (batch_size, vector_size)
            margin:     margin大小， scalar
            
        Returns:
            triplet_loss: scalar, 一个batch的损失值
            fraction_postive_triplets : valid的triplets占的比例
    '''
    # 得到每两两embeddings的距离，然后增加一个维度，一维需要得到（batch_size, batch_size, batch_size）大小的3D矩阵
    # 然后再点乘上valid 的 mask即可
    pairwise_dis = pairwise_distance(embed1,embed2)
    anchor_positive_dist = torch.unsqueeze(pairwise_dis, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = torch.unsqueeze(pairwise_dis, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = triplet_mask(labels)
    mask = mask.float()
    triplet_loss = triplet_loss.to(device)
    triplet_loss = torch.multiply(mask, triplet_loss)
    #print(triplet_loss[:100])
    triplet_loss = torch.clamp(triplet_loss, 0.0)
    #print(triplet_loss)
    # 计算valid的triplet的个数，然后对所有的triplet loss求平均
    valid_triplets = torch.greater(triplet_loss, 1e-16).float()
    #print(valid_triplets)
    num_positive_triplets = torch.sum(valid_triplets)
    print("tp,numpos: ",torch.sum(triplet_loss),torch.sum(triplet_loss).dtype,num_positive_triplets)
    #print(num_positive_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    #print(triplet_loss.shape)
    
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    #print(type(triplet_loss))
    triplet_loss = triplet_loss.requires_grad_()
    #print(triplet_loss)
    #print(triplet_loss.shape)
    print("numvalid: ",num_valid_triplets)
    return triplet_loss.to(device), fraction_postive_triplets
'''
ll = torch.tensor([1,1,3,3,5])
tm = triplet_mask(ll)
print(tm)
em = torch.tensor([[1,1],[8,8],[2,6],[5,3],[2,2]])
print(pairwise_distance(em,em))
tl,fc = batch_all_triplet_loss(ll,em,em,0)
print(tl.item())

sss = np.array([[1],[0],[3],[2],[]])    
print(type(sss[0])) 
fff = tp_loss(sss,em,em,0,5,0) 
print(fff)
'''

def process_sameid(samid,rg):
    ct = 0
    s = []
    for i in range(rg):
        if i % 2 == 0:

            app = [i,i+1]
        s.append(np.append(samid[i],app))
    return s

def tp_loss(sameid,em1,em2,margin,bts,ith):
    pair_dist = pairwise_distance(em1,em2)
    #print(bts,ith)
    tloss = 0.0
    validate = 0
    compl = set(np.arange(bts+15))
    fff = 0
    val = 0
    cvv = 0
    for k in range(bts):
        sid = sameid[k]
        sid = np.array(sid)
        s1 = set(list(sid))
        nsi = list(compl-s1)
        negs = pair_dist[k][nsi]

        for p in sid:
            if p != k:
                if p <  min((ith+1)*bts,68120) and p != k:
                    #print(p)
                    p = p % bts
                    cvv += 1
                else:
                    p = p%bts + bts
                triplet = torch.clamp(pair_dist[k][p]-negs+margin,min=0.0)
                val += torch.sum(torch.greater(triplet, 1e-16).float())
                #triplet = torch.clamp(-negs+margin,min=0.0)
                validate += torch.count_nonzero(triplet)
                tloss += torch.sum(triplet,dtype=torch.float32)
                    #print(tloss,validate)
            
    #print("MMMMM: ",tloss,validate,val,cvv)
    if validate == 0:
        return 0
    return tloss/validate

"""
bb = 800
c_loss2 = tp_loss(process_sameid(same_id,bb),encoded_img[:bb],encoded_text[:bb],50,bb,0)

c_loss,_ = batch_all_triplet_loss(label[:bb],encoded_img[:bb],encoded_text[:bb],50)
print(c_loss2,c_loss)
"""
def tp_loss_hardest(sameid,em1,em2,margin,bts):
    pair_dist = pairwise_distance(em1,em2)
    compl = set(np.arange(bts))
    tloss = 0
    validate = 0
    for k in range(bts):
        sid = sameid[k]
        sid = sid%bts
        #if len(sid) == 0:
            #break
        posi = np.array(sid)
        hardest_pos = torch.max(pair_dist[k][posi])
        s1 = set(list(posi)+[k])
        negt = list(compl-s1)
        hardest_neg = torch.min(pair_dist[k][negt])
        triplet = torch.clamp(hardest_pos-hardest_neg+margin,min=0)
        if triplet.item() != 0:
            validate += 1
        tloss += triplet
    return tloss/validate

##maximum = getpartindex(same_id,1024)
#print(len(maximum),68120//1024) 
def train(e_num,mn,b,lr1,lr2,batch_size,prt,eimg,etext,image_features,text_features,linear_model2,linear_model3):
    num_epochs = e_num
    epoch_iterator = trange(num_epochs)
    loss_fn = nn.CrossEntropyLoss()
    num_epo = 0
    #optimizer = optim.Adam(linear_model3.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-9)
    #optimizer = optim.SGD(linear_model3.parameters(), lr=0.005, momentum=0.9)
    linear_model2 = linear_model2.to(device)
    linear_model3 = linear_model3.to(device)
    #print(next(linear_model2.parameters()).device)
    
    
    dimg,dtext,rdim = modelinfodict[mn]
    print(dimg,dtext,rdim)
    '''
    linear_model4 = ft.FullyConnected_dropout(rdim,3072,2048,rdim,0.67,0.15,0.15)
    linear_model5 = ft.FullyConnected_dropout(rdim,3072,2048,rdim,0.77,0.15,0.15)
    timage = torch.load("encode/"+dimg+"test_img"+".pt")
    ttext = torch.load("encode/"+dtext+"test_text"+".pt")
    #print(image_features.shape,text_features.shape)
    encoded_img,encoded_text = load_data(rdim,dimg,dtext)
    
    optimizer = optim.AdamW([
        {'params':linear_model3.parameters(), 'lr':lr1, 'betas':(0.9,0.95),'weight_decay':1e-3},
        {'params':linear_model2.parameters(), 'lr':lr2, 'betas':(0.9,0.95),'weight_decay':1e-3},
        {'params':linear_model4.parameters(), 'lr':lr2, 'betas':(0.9,0.95),'weight_decay':1e-3},
        {'params':linear_model5.parameters(), 'lr':lr2, 'betas':(0.9,0.95),'weight_decay':1e-3}
    ])
    '''
    optimizer = optim.Adam([
        {'params':linear_model3.parameters(), 'lr':lr1, 'betas':(0.9,0.95),'weight_decay':1e-3},
        {'params':linear_model2.parameters(), 'lr':lr2, 'betas':(0.9,0.95),'weight_decay':1e-3},
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
    label = test_data_name['id']
    same_id = test_data_name['same_id_index']
    same_id = np.array(same_id)
    best_test_r1 = 0
    best_test_map = 0
    lab = np.array(label)
    tr_num = eimg.shape[0]
    #schedular = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.93)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.5, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    txind = np.array([2*i+num_epo%2 for i in range(34060)])
    same_id = process_sameid(same_id,68120)
    same_id = np.array(same_id)
    sele = np.arange(68120)
    auges = ['r','v','s','e','c','h']
    coeff = [2,2,2,2,2,2]

    #3covo = ft.ConvNet()

    for epoch in epoch_iterator:
        np.random.shuffle(sele)
        for i in range(train_data_num//batch_size):
            linear_model3.train()
            linear_model2.train()
            #covo.train()
            linear_model2 = linear_model2.to(device)
            linear_model3 = linear_model3.to(device)
            #covo = covo.to(device)

            #linear_model4.train()
            #linear_model5.train()
            #linear_model4 = linear_model2.to(device)
            #linear_model5 = linear_model3.to(device)
            #if2,tf2 = getbatch_data(encoded_img,encoded_text,batch_size,i*batch_size,device)
            #if2 = encoded_img[sele[i*batch_size:min(i*batch_size+15+batch_size,eimg.shape[0])]]
            #tf2 = encoded_text[sele[i*batch_size:min(i*batch_size+15+batch_size,eimg.shape[0])]]
            #image_feature,text_feature = getbatch_data(eimg,etext,batch_size,i*batch_size,device)
            #labels = get_label(label,batch_size,i*batch_size)
            image_feature= eimg[sele[i*batch_size:min(i*batch_size+15+batch_size,eimg.shape[0])]]
            text_feature = etext[sele[i*batch_size:min(i*batch_size+15+batch_size,eimg.shape[0])]]
            
            #image_feature2,text_feature2 = getbatch_data(eigr,etxr,batch_size,i*batch_size,device)
            #labels = get_label(label,batch_size,i*batch_size)
            if i == 0 and num_epo == 0:
                #print("SSSSS")
                with torch.no_grad():
                    imag = image_features/image_features.norm(dim=1, keepdim=True)
                    txt = text_features/text_features.norm(dim=1, keepdim=True)
                    logits_per_image = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) * imag @ txt.t()
                    logits_per_text = logits_per_image.t()
                    probs = logits_per_text.softmax(dim=-1).detach().cpu().numpy()
                    #print(probs.shape)
                    t2i_cmc, t2i_map = cal.evaluate(probs, txt_labels, img_labels)
                    strr = "t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, map: {:.4}".format(t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_map)
                    print(strr)
            
            labels = np.arange(image_feature.shape[0])
            
            #sam = get_label(same_id,batch_size,i*batch_size)
            sam = get_label(same_id,batch_size,i*batch_size)
            #labels = get_label(cap_match_imgind,batch_size,i*batch_size)
            # normalized features
            c_loss = 0
            fc_img1 = linear_model3(image_feature.float())
            fc_text1 = linear_model2(text_feature.float())
            #for i in range(batch_size):
                #ig = Image.open(path)
                #all images
            #ig_cov_features = covo(xxx)
            #fc_img1 += ig_cov_features
            #fi,fte = linear_model4(if2.float()),linear_model5(tf2.float())
            
            #c_loss,fcloss = batch_all_triplet_loss(get_label(lab,batch_size,i*batch_size),fc_img1,fc_text1,15)
            bts = image_feature.shape[0]
            #c_loss = tp_loss(sam,fc_img1,fc_text1,45,bts)
            #c_loss2 = tp_loss( sam,fc_img1[:],fc_text1[:],15,batch_size,i)
            #c_loss2 = tp_loss( sam,fc_img1[:],fc_img1[:],7,batch_size,i)
            #c_loss2 += tp_loss( sam,fc_text1[:],fc_text1[:],7,batch_size,i)
            #c_loss,_ = batch_all_triplet_loss(label[i*batch_size:min(68120,(i+1)*batch_size)],fc_img1[:batch_size],fc_text1[:batch_size],50)
            fc_img1 = fc_img1 / fc_img1.norm(dim=1, keepdim=True)
            fc_text1 = fc_text1 / fc_text1.norm(dim=1, keepdim=True)
            #fi = fi/fi.norm(dim=1, keepdim=True)
            #fte = fte/fte.norm(dim=1, keepdim=True)
            #fc_img1 += fi
            #fc_text1 += fte

            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            logit_scales = logit_scale.exp()

            #logits_per_image3 = fi @ fte.t()
            #logits_per_image3 *= logit_scales
            #logits_per_text3 = logits_per_image3.t()

            logits_per_image1 = fc_img1 @ fc_text1.t()
            logits_per_image1 *= logit_scales
            logits_per_text1 = logits_per_image1.t()

            #logits_per_image1 += logits_per_image3
            #logits_per_text1 += logits_per_text3

            loss_img = loss_fn(logits_per_image1.to(torch_device),torch.tensor(labels).to(torch_device).long())
            loss_text = loss_fn(logits_per_text1.to(torch_device),torch.tensor(labels).to(torch_device).long())
            loss = (loss_img + loss_text)/2 ######/2
            for a,co in zip(auges,coeff):
                ex_encimg = load_img(rdim,dimg,dtext,a)
                image_feature = ex_encimg[sele[i*batch_size:min(i*batch_size+15+batch_size,eimg.shape[0])]]
                fc_img1 = linear_model3(image_feature.float())
                fc_img1 = fc_img1 / fc_img1.norm(dim=1, keepdim=True)

                logits_per_image1 = fc_img1 @ fc_text1.t()
                logits_per_image1 *= logit_scales
                logits_per_text1 = logits_per_image1.t()
            
                loss_img3 = loss_fn(logits_per_image1.to(torch_device),torch.tensor(labels).to(torch_device).long())
                loss_text3 = loss_fn(logits_per_text1.to(torch_device),torch.tensor(labels).to(torch_device).long())
                loss += co*(loss_img3 + loss_text3)/2
               
            if num_epo > 20000:
                c_loss = tp_loss_hardest(sam,fc_img1,fc_img1,45,bts)
                c_loss += tp_loss_hardest(sam,fc_text1,fc_text1,45,bts)
                loss = loss + 0.2*c_loss
            #loss += 0.8*c_loss2
            loss /= len(auges)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % prt == 0:
                with torch.no_grad():
                    #param1 = list(linear_model3.named_parameters())
                    #print("alpha: ",param1[0][1].item())
                    #param2 = list(linear_model2.named_parameters())
                    #print("beta: ",param2[0][1].item())
                    linear_model2 = linear_model2.cuda()
                    linear_model3 = linear_model3.cuda()
                    fc_img2 = linear_model3(image_features.float())
                    fc_text2 = linear_model2(text_features.float())
                    fc_img2 = fc_img2/fc_img2.norm(dim=1, keepdim=True)
                    fc_text2 = fc_text2/fc_text2.norm(dim=1, keepdim=True)

                    logits_per_image2 = logit_scales * fc_img2 @ fc_text2.t()
                    logits_per_text2 = logits_per_image2.t()
                    #logits_per_text2 += (logit_scales * fti @ ftt.t()).t()
                    
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
        schedular.step(t2i_cmc[0])
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

'''
load the test data
'''

modelname = 'ViT-L/14'
modelinfodict = {"RN50x64":["img2/","text2/",1024],"RN50x16":["img3/","text3/",768],
                "RN50x4":["img4/","text4/",640],"RN101":["img5/","text5/",512],
                "ViT-L/14@336px":["img6/","text6/",768],"ViT-B/32":["img7/","text7/",512],
                "ViT-B/16":["img8/","text8/",512],"ViT-L/14":["img/","text/",768]}
dimg,dtext,rdim = modelinfodict[modelname]
print(dimg,dtext,rdim)

image_features = torch.load("encode/"+dimg+"test_img"+".pt")
text_features = torch.load("encode/"+dtext+"test_text"+".pt")
print(image_features.shape,text_features.shape)
encoded_img,encoded_text = load_data(rdim,dimg,dtext)


def Residual(modelnames):
    tps = [modelinfodict[mn] for mn in modelnames]
    te_fusion_img,te_fusion_text,tr_fusion_img,tr_fusion_text = [],[],[],[]
    for tp in tps:
        di,dt,dim = tp
        te_fusion_img.append(torch.load("encode/"+di+"test_img"+".pt"))
        te_fusion_text.append(torch.load("encode/"+dt+"test_text"+".pt")) 
        trimg,trtext = load_data(dim,di,dt)
        tr_fusion_img.append(trimg);tr_fusion_text.append(trtext)
    #print(torch.sum(torch.stack(te_fusion_img),dim=0).shape)
    return torch.sum(torch.stack(te_fusion_img),dim=0),torch.sum(torch.stack(te_fusion_text),dim=0),torch.sum(torch.stack(tr_fusion_img),dim=0),torch.sum(torch.stack(tr_fusion_text),dim=0)



def cancatenate(modelnames):
    tps = [modelinfodict[mn] for mn in modelnames]
    te_fusion_img,te_fusion_text,tr_fusion_img,tr_fusion_text = [],[],[],[]
    for tp in tps:
        di,dt,dim = tp
        te_fusion_img.append(torch.load("encode/"+di+"test_img"+".pt"))
        te_fusion_text.append(torch.load("encode/"+dt+"test_text"+".pt")) 
        trimg,trtext = load_data(dim,di,dt)
        tr_fusion_img.append(trimg);tr_fusion_text.append(trtext)
    #print(torch.sum(torch.stack(te_fusion_img),dim=0).shape)
    return torch.cat(te_fusion_img,dim=1),torch.cat(te_fusion_text,dim=1),torch.cat(tr_fusion_img,dim=1),torch.cat(tr_fusion_text,dim=1)


F_te_i,F_te_t,F_enci,F_enct = Residual(["RN50x16","ViT-L/14",'ViT-L/14@336px'])
F_te_i,F_te_t,F_enci,F_enct = Residual(["ViT-L/14",'ViT-L/14@336px'])
F_te_i,F_te_t,F_enci,F_enct = Residual(["ViT-B/16",'ViT-B/32'])
F_te_i,F_te_t,F_enci,F_enct = cancatenate(["ViT-B/16","RN101"])
print(F_te_i.shape,F_te_t.shape,F_enci.shape,F_enct.shape)

linear_model2 = ft.FullyConnected_dropout(1536,3072,2048,1536,0.17,0.25,0.25)
linear_model3 = ft.FullyConnected_dropout(1536,3072,2048,1536,0.27,0.25,0.25)

linear_model2 = ft.FullyConnected_dropout(1024,3072,2048,1024,0.17,0.1,0.1)
linear_model3 = ft.FullyConnected_dropout(1024,3072,2048,1024,0.27,0.1,0.1)
'''RN50*64 1024'''
linear_model2 = ft.FullyConnected_dropout(rdim,4096,2048,rdim,0.17,0.15,0.15)
linear_model3 = ft.FullyConnected_dropout(rdim,4096,2048,rdim,0.27,0.15,0.15)
linear_model2 = ft.FullyConnected2(rdim,3072,rdim,0.55)
linear_model3 = ft.FullyConnected2(rdim,3072,rdim,0.78)
linear_model2 = ft.FullyConnectedApp(rdim,3072,rdim,232)
linear_model3 = ft.FullyConnectedApp(rdim,3072,rdim,232)
linear_model2 = ft.FClayer(rdim,4096,2048,rdim,0.4)
linear_model3 = ft.FClayer(rdim,4096,2048,rdim,0.4)


'''RN50*4 640'''

linear_model2 = ft.FullyConnected_dropout(rdim,3072,2048,rdim,0.17,0.0,0.0)
linear_model3 = ft.FullyConnected_dropout(rdim,3072,2048,rdim,0.17,0.0,0.0)
linear_model2 = ft.FullyConnected2(rdim,3072,rdim,0.55)
linear_model3 = ft.FullyConnected2(rdim,3072,rdim,0.78)
linear_model2 = ft.FullyConnectedApp(rdim,3072,rdim,232)
linear_model3 = ft.FullyConnectedApp(rdim,3072,rdim,232)
linear_model2 = ft.FClayer(rdim,3072,2048,rdim,0.5)
linear_model3 = ft.FClayer(rdim,3072,2048,rdim,0.5)


'''RN101 '''
rdim = 768
linear_model2 = ft.FullyConnected_dropout(rdim,3072,2048,rdim,0.17,0.05,0.05)
linear_model3 = ft.FullyConnected_dropout(rdim,3072,2048,rdim,0.27,0.05,0.05)
'''Vit-L/14 768'''
'''ViT-L/14@336px '''
'''RN50*16 768'''
rdim = 768
linear_model2 = ft.FullyConnected_dropout(rdim,3072,2048,rdim,0.67,0.05,0.05)
linear_model3 = ft.FullyConnected_dropout(rdim,3072,2048,rdim,0.77,0.05,0.05)
linear_model2 = ft.FullyConnected_dropout2(rdim,3072,2048,rdim,0.75,0.0,0.0)
linear_model3 = ft.FullyConnected_dropout2(rdim,3072,2048,rdim,0.75,0.0,0.0)
linear_model2 = ft.FullyConnected2(rdim,3072,rdim,0.55)
linear_model3 = ft.FullyConnected2(rdim,3072,rdim,0.78)
linear_model2 = ft.FullyConnectedApp(rdim,3072,rdim,232)
linear_model3 = ft.FullyConnectedApp(rdim,3072,rdim,232)
linear_model2 = ft.FClayer(rdim,3072,2048,rdim,1.0,0.3,0.3)
linear_model3 = ft.FClayer(rdim,3072,2048,rdim,1.0,0.3,0.3)


train(65,"ViT-L/14",0.2,11e-5,11e-5,1024,15,encoded_img,encoded_text,image_features,text_features,linear_model2,linear_model3)

train(100,"ViT-L/14",0.2,20e-5,20e-5,2048,20,F_enci,F_enct,F_te_i,F_te_t,linear_model2,linear_model3)

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