from __future__ import print_function
import argparse
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_regdb_debug, eval_sysu_debug
from models.model import embed_net
from models.rga_model import embed_net_rga
# from models.model_debug import embed_net_debug           # vis pool feature distribution on original images

from utils import *
import time 
import random
import scipy.io as scio
import Transform as transforms


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--visualization', '-v', default=False, type=bool, help='visualization retrieval result')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--per_img', default=8, type=int,
                    help='number of samples of an id in every batch')
parser.add_argument('--w_hc', default=0.5, type=float,
                    help='weight of Hetero-Center Loss')
parser.add_argument('--gall-mode', default='single', type=str, help='single or multi')

args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = './data/sysu/'
    n_class = 395
    test_mode = [1, 2] 
elif dataset =='regdb':
    data_path = '../IVReIDData/RegDB/'
    n_class = 206
    test_mode = [2, 1]
    
best_acc = 0  # best test accuracy
start_epoch = 0 

if args.visualization:
    bad_thre = 0.5
    num_id_to_draw = 10

print('==> Building model..')
# net = embed_net(args.low_dim, n_class, drop=0.0, arch=args.arch)
# net = embed_net_debug(args.low_dim, n_class, drop=0.0, arch=args.arch)                # draw pool features
net = embed_net_rga(args.low_dim, n_class, height=args.img_h, width=args.img_w, pretrained=True, dropout=args.drop, branch_name='rgas')

net.cuda()    
cudnn.benchmark = True

print('==> Loading model..')
if len(args.resume)>0:   
    model_path = 'save_model/' + args.resume
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> [Error] no checkpoint found at {}!!!'.format(args.resume))
        sys.exit()

print('==> Loading data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize((args.img_h,args.img_w)),
    transforms.RectScale(args.img_h, args.img_w),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset =='sysu':
    query_img_path, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img_path, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, gall_mode=args.gall_mode)

elif dataset =='regdb':
    query_img_path, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    gall_img_path, gall_label = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')
    
    gallset = TestData(gall_img_path, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
nquery = len(query_label)
ngall = len(gall_label)
print("  ------------------------------")
print("  Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")

queryset = TestData(query_img_path, query_label, transform = transform_test, img_size=(args.img_w, args.img_h))   
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))

feature_dim = args.low_dim

if args.arch =='resnet50':
    pool_dim = 2048
elif args.arch =='resnet18':
    pool_dim = 512

def extract_feat_debug(data_loader,data_num,forward_mode):
    net.eval()

    start = time.time()
    ptr = 0
    
    pool_feats = np.zeros((data_num, 2048, 18, 9))
    with torch.no_grad():
        for _, (input, _) in enumerate(data_loader):
            batch_num = input.size(0)
            input = input.cuda()
            pool_feat = net(input, input, forward_mode)

            pool_feats[ptr:ptr+batch_num,: ] = pool_feat.detach().cpu().numpy()
            ptr += batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return pool_feats

def extract_feat(data_loader,data_num,forward_mode):
    net.eval()

    start = time.time()
    ptr = 0
    feats = np.zeros((data_num, 6*feature_dim))
    feat_pools = np.zeros((data_num, 6*pool_dim))
    with torch.no_grad():
        for _, (input, _) in enumerate(data_loader):
            batch_num = input.size(0)
            input = input.cuda()
            pool_feat, feat = net(input, input, forward_mode)
            feat_pools[ptr:ptr+batch_num,: ] = pool_feat.detach().cpu().numpy()
            feats[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr += batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return feats, feat_pools 

def save_feat(numpy_feat,save_path):
    np.save(save_path,numpy_feat) 
    print("save to ",save_path)

def draw_rect(img,color_mode):
    img = np.array(img)
    rects = [(0, 0, img.shape[1], img.shape[0])]
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), color_mode, 2)
    return img

def draw_retri_images_regdb(distmat, save_path, draw_id_list=None, bad_match_ids=None, num_id_to_draw=5, top_k_to_plot=10):
    num_q = distmat.shape[0]
    
    if draw_id_list is None:
        pool = np.arange(num_q,dtype=int)

        if bad_match_ids is not None:
            pool = np.delete(pool,bad_match_ids)
            print("removing bad_match_ids")

        draw_id_list = np.random.choice(pool,num_id_to_draw,replace=False)
    else:
        if len(draw_id_list) > num_id_to_draw:
            draw_id_list = random.sample(draw_id_list,num_id_to_draw)

    print("draw_id_list: ",draw_id_list)
    indices = np.argsort(distmat, axis=1)

    fig=plt.figure(figsize=(8, 8))
    row = len(draw_id_list)
    column = top_k_to_plot+1 
    for qi, qidx in enumerate(draw_id_list):
        q_img_path = query_img_path[qidx]
        q_img = Image.open(q_img_path)
        
        print("------")
        print("query label, ",query_label[qidx])
        
        fig.add_subplot(row,column,qi*(top_k_to_plot+1)+1)
        plt.imshow(q_img)
        plt.axis('off')
        
        order = indices[qidx,:]
        top_k_g_labels = gall_label[order][:top_k_to_plot]
        print("gallery labels (top k), ",top_k_g_labels)
        g_img_paths = [gall_img_path[i] for i in order[:top_k_to_plot]]

        for gi, g_img_path in enumerate(g_img_paths):
            g_img = Image.open(g_img_path)
            
            if top_k_g_labels[gi] != query_label[qidx]:
                color_mode = (255, 0, 0)            # wrong
            else:
                color_mode = (0, 255, 0)            # right
            g_img = draw_rect(g_img, color_mode)

            fig.add_subplot(row,column,qi*(top_k_to_plot+1)+2+gi)
            plt.imshow(g_img)
            plt.axis('off')

    plt.savefig(save_path)
    print("save to ",save_path)

def draw_retri_images_sysu(bad_q_labels, bad_g_labels, bad_q_paths, bad_g_paths, save_path, num_id_to_draw=5, top_k_to_plot=10, shuffle_draw=True):
    
    if shuffle_draw:
        if num_id_to_draw < len(bad_q_paths):
            draw_id_list = np.random.choice(len(bad_q_paths),num_id_to_draw,replace=False)
    else:
        draw_id_list = np.arange(num_id_to_draw)

    print("draw_id_list: ",draw_id_list)

    fig=plt.figure(figsize=(50, 50))
    row = len(draw_id_list)
    column = top_k_to_plot+1 
    for qi, qidx in enumerate(draw_id_list):
        q_img_path = bad_q_paths[qidx]
        q_img = Image.open(q_img_path)
        
        print("------")
        q_label = bad_q_labels[qidx]
        print("query label, ",q_label)
        
        fig.add_subplot(row,column,qi*(top_k_to_plot+1)+1)
        plt.imshow(q_img)
        plt.axis('off')

        top_k_g_labels = bad_g_labels[qidx][:top_k_to_plot]
        print("gallery labels (top k), ",top_k_g_labels)
        top_k_g_paths = bad_g_paths[qidx][:top_k_to_plot]

        for gi, g_path in enumerate(top_k_g_paths):
            g_img = Image.open(g_path)
            
            if top_k_g_labels[gi] != q_label:
                color_mode = (255, 0, 0)            # wrong
            else:
                color_mode = (0, 255, 0)            # right
                print("get one hit!")
            g_img = draw_rect(g_img, color_mode)

            fig.add_subplot(row,column,qi*(top_k_to_plot+1)+2+gi)
            plt.imshow(g_img)
            plt.axis('off')

    plt.savefig(save_path)
    print("save to ",save_path)

def ReadBadIndex(txt_path):
    with open(txt_path,'r') as f:
        lines = f.readlines()
    lines = lines[0]
    lines = lines.strip().split()
    line = [int(i) for i in lines]

    return line

query_feat, query_feat_pool = extract_feat(query_loader,nquery,test_mode[1])    
# query_feat_pool = extract_feat_debug(query_loader,nquery,test_mode[1])  

all_cmc = 0
all_mAP = 0 
all_cmc_pool = 0

if dataset =='regdb':
    gall_feat, gall_feat_pool = extract_feat(gall_loader,ngall,test_mode[0])
    # gall_feat_pool = extract_feat_debug(gall_loader,ngall,test_mode[0])
    # save_feat(query_feat_pool,'query_feat_pool.npy')
    
    ##### -------- fc feature 
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    if args.visualization:
        cmc, mAP, bad_match_ids = eval_regdb_debug(-distmat, query_label, gall_label, bad_thre, max_rank=20, write_bad_to_txt=False)
        print("ratio of bad_match_id(top50, thre{}): {}/{}".format(bad_thre,len(bad_match_ids),nquery))  
        
        # bad_match_ids = None
        # bad_match_ids = ReadBadIndex('same_id_index.txt')
        # bad_match_ids = bad_match_ids[10:20]

        draw_retri_images(-distmat,draw_id_list=bad_match_ids,bad_match_ids=None,save_path='./result/{}ts_same_bad_case_10-20.pdf'.format(dataset),num_id_to_draw=num_id_to_draw)                    # draw selected bad matches

        # draw_retri_images(-distmat,draw_id_list=bad_match_ids,bad_match_ids=None,save_path='./result/badcase_top50_thre{}_mAP{:.2f}.pdf'.format(bad_thre,mAP*100),num_id_to_draw=num_id_to_draw)                    # draw bad matched queries
        
        # draw_retri_images(-distmat,draw_id_list=None,bad_match_ids=None,save_path='./result/result_rand_nid{}.pdf'.format(num_id_to_draw),num_id_to_draw=num_id_to_draw)                                # draw random queries
    else:
        cmc, mAP = eval_regdb(-distmat, query_label, gall_label, max_rank=20)

    print("==== Result ====")
    print ('Test Trial: {}'.format(args.trial))
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))

    ##### -------- pool5 feature
    # distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    # cmc_pool, mAP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

    # print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
    #     cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    # print('mAP: {:.2%}'.format(mAP_pool))

    
elif dataset =='sysu':
    n_trial = 5
    for trial in range(n_trial):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, gall_mode=args.gall_mode)
        
        trial_gallset = TestData(gall_img, gall_label, transform = transform_test,img_size=(args.img_w,args.img_h))
        trial_gall_loader  = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        
        gall_feat, gall_feat_pool = extract_feat(trial_gall_loader,ngall,test_mode[0])
        
        # fc feature 
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        if args.visualization:
            cmc, mAP, bad_match_ids, bad_q_labels, bad_g_labels, bad_q_paths, bad_g_paths = eval_sysu_debug(-distmat, query_label, gall_label, query_cam, gall_cam, query_img_path, gall_img_path, write_bad_to_txt=False)

            draw_retri_images_sysu(bad_q_labels, bad_g_labels,bad_q_paths,bad_g_paths,save_path='./result/{}_badcase_1hitAfter10_mAP{:.2f}_trial{}.pdf'.format(dataset,mAP*100,trial),num_id_to_draw=num_id_to_draw)                    # draw selected bad matches
        else:
            cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        
        ## *********** pool5 feature **********
        # distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        # cmc_pool, mAP_pool = eval_sysu(-distmat_pool, query_label, gall_label,query_cam, gall_cam)
        if trial==0:
            all_cmc = cmc
            all_mAP = mAP
            # all_cmc_pool = cmc_pool
            # all_mAP_pool = mAP_pool
        else:
            all_cmc += cmc
            all_mAP += mAP
            # all_cmc_pool += cmc_pool
            # all_mAP_pool += mAP_pool
        
        print ('*'*10,'Test Trial: {}'.format(trial),'*'*10)
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
        # print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            # cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
        # print('mAP: {:.2%}'.format(mAP_pool))


    cmc = all_cmc /n_trial
    mAP = all_mAP /n_trial

    # cmc_pool = all_cmc_pool /10 
    # mAP_pool = all_mAP_pool /10
    print ('*'*10,'All Average:','*'*10)
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))

    # print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
    #     cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    # print('mAP: {:.2%}'.format(mAP_pool))