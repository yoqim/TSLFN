from __future__ import print_function
import argparse
import sys
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
# from model_debug import embed_net_debug           # vis pool feature distribution on original images

from utils import *
import time 
import scipy.io as scio
import Transform as transforms


#python test.py --dataset regdb --trial 1 --gpu 1 --low-dim 512 --resume 'regdb_id_bn_relu_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_l2_md_all_lossback_trial_1_best.t' --w_hc 0.5


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
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

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# np.random.seed(1)
# random.seed(1)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../IVReIDData/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2] 
elif dataset =='regdb':
    data_path = '../IVReIDData/RegDB/'
    n_class = 206
    test_mode = [2, 1]
    
best_acc = 0  # best test accuracy
start_epoch = 0 

print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop=0.0, arch=args.arch)
# net = embed_net_debug(args.low_dim, n_class, drop=0.0, arch=args.arch)
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
        print('==> no checkpoint found at {}'.format(args.resume))

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

def draw_retri_images(distmat, save_path, num_id_to_draw=5, top_k_to_plot=10):
    num_q = distmat.shape[0]
    indices = np.argsort(distmat, axis=1)
    draw_id_list = np.random.choice(num_q,num_id_to_draw,replace=False)
    print("draw_id_list: ",draw_id_list)

    fig=plt.figure(figsize=(8, 8))
    row = num_id_to_draw
    column = top_k_to_plot+1 
    for qi, qid in enumerate(draw_id_list):
        q_img_path = query_img_path[qid]
        q_img = Image.open(q_img_path)

        fig.add_subplot(row,column,qi*(top_k_to_plot+1)+1)
        plt.imshow(q_img)
        plt.axis('off')
        
        order = indices[qid,:]
        g_img_paths = [gall_img_path[i] for i in order[:top_k_to_plot]]
        
        for gi, g_img_path in enumerate(g_img_paths):
            g_img = Image.open(g_img_path)
            fig.add_subplot(row,column,qi*(top_k_to_plot+1)+2+gi)
            plt.imshow(g_img)
            plt.axis('off')

    plt.savefig(save_path)
    print("save to ",save_path)



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
    cmc, mAP = eval_regdb(-distmat, query_label, gall_label,max_rank=20)

    draw_retri_images(distmat,save_path='result.pdf',num_id_to_draw=5)
    
    import pdb;pdb.set_trace()
    print("query_feat shape, ", query_feat.shape)
    print("gall_feat shape, ", gall_feat.shape)
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
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, gall_mode=args.gall_mode)
        
        trial_gallset = TestData(gall_img, gall_label, transform = transform_test,img_size =(args.img_w,args.img_h))
        trial_gall_loader  = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        
        gall_feat, gall_feat_pool = extract_feat(trial_gall_loader,ngall,test_mode[0])
        
        # fc feature 
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        cmc, mAP  = eval_sysu(-distmat, query_label, gall_label,query_cam, gall_cam)
        
        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool = eval_sysu(-distmat_pool, query_label, gall_label,query_cam, gall_cam)
        if trial ==0:
            all_cmc = cmc
            all_mAP = mAP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
        
        print ('Test Trial: {}'.format(trial))
        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
        print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
        print('mAP: {:.2%}'.format(mAP_pool))

    cmc = all_cmc /10 
    mAP = all_mAP /10

    cmc_pool = all_cmc_pool /10 
    mAP_pool = all_mAP_pool /10
    print ('All Average:')
    print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[4], cmc[9], cmc[19]))
    print('mAP: {:.2%}'.format(mAP))
    print('POOL5: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}'.format(mAP_pool))