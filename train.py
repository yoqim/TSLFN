from __future__ import print_function
import argparse
import sys
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from models.model import embed_net
from models.rga_model import embed_net_rga
from utils import *
import Transform as transforms
from heterogeneity_loss import hetero_loss
from triplet_loss import OriTripletLoss
import xlwt,xlrd


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, 
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, 
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only') 
parser.add_argument('--model_path', default='save_model/', type=str, 
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, 
                    help='log save path')
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
parser.add_argument('--method', default='id', type=str,
                    metavar='m', help='method type')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--per_img', default=8, type=int,
                    help='number of samples of an id in every batch')
parser.add_argument('--w_hc', default=0.5, type=float,
                    help='weight of Hetero-Center Loss')
parser.add_argument('--thd', default=0, type=float,
                    help='threshold of Hetero-Center Loss')
parser.add_argument('--epochs', default=500, type=int,
                    help='weight of Hetero-Center Loss')
parser.add_argument('--dist-type', default='l2', type=str,
                    help='type of distance')


torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)

def worker_init_fn(worker_id):
    # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(0 + worker_id)
    
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    
dataset = args.dataset
if dataset == 'sysu':
    data_path = './data/sysu/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]                          # thermal to visible
elif dataset =='regdb':
    data_path = '../IVReIDData/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]                          # visible to thermal

checkpoint_path = args.model_path + args.dataset + '/'

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
 
if args.method =='id':
    suffix = dataset + '_id_bn_relu'
suffix = suffix + '_lr_{:1.1e}'.format(args.lr) 
suffix = suffix + '_md_{}'.format(args.mode)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim
if dataset =='regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

suffix += 'RGBs_4_wclsbias'
# suffix += '_wclsbias'                       

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path + suffix + '_os.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_mAP = 0
start_epoch = 0 
feature_dim = args.low_dim

print('==> Loading data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RectScale(args.img_h, args.img_w),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RectScale(args.img_h, args.img_w),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset =='sysu':
    loss_print_interval = 100
    # training set
    trainset = SYSUData(data_path,  transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img_path, query_label, query_cam = process_query_sysu(data_path, mode = args.mode)
    gall_img_path, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode)
      
elif dataset =='regdb':
    loss_print_interval = 20
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img_path, query_label = process_test_regdb(data_path, trial = args.trial, modal = 'visible')
    gall_img_path, gall_label  = process_test_regdb(data_path, trial = args.trial, modal = 'thermal')

gallset = TestData(gall_img_path, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
queryset = TestData(query_img_path, query_label, transform = transform_test, img_size =(args.img_w,args.img_h))
    
# testing data loader
gall_loader  = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, worker_init_fn=worker_init_fn)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, worker_init_fn=worker_init_fn)
   
n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('  Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')   
print('  Data Loading Time:\t {:.3f}'.format(time.time()-end))

print('==> Building model...')
# net = embed_net(args.low_dim, n_class, drop=args.drop, arch=args.arch)
net = embed_net_rga(args.low_dim, n_class, height=args.img_h, width=args.img_w, pretrained=True, dropout=args.drop, branch_name='rgas')

net.to(device)
cudnn.benchmark = True

if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

if args.method =='id':
    thd = args.thd
    criterion = nn.CrossEntropyLoss()
    criterion_het = hetero_loss(margin=thd, dist_type=args.dist_type)
    # criterion_tri = OriTripletLoss(margin=0.3)

    criterion.to(device)
    criterion_het.to(device)
    # criterion_tri.to(device)

## setting different lr to baseline / classifiers
def set_ignored_params(net):
    ignored_params = list(map(id, net.feature1.parameters())) \
                    + list(map(id, net.feature2.parameters())) \
                    + list(map(id, net.feature3.parameters())) \
                    + list(map(id, net.feature4.parameters())) \
                    + list(map(id, net.feature5.parameters())) \
                    + list(map(id, net.feature6.parameters())) \
                    + list(map(id, net.classifier1.parameters())) \
                    + list(map(id, net.classifier2.parameters())) \
                    + list(map(id, net.classifier3.parameters()))\
                    + list(map(id, net.classifier4.parameters()))\
                    + list(map(id, net.classifier5.parameters()))\
                    + list(map(id, net.classifier6.parameters()))
    return ignored_params

def set_optimizer(optim_name,net,base_params):
    if optim_name == 'sgd':
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1*args.lr},
            {'params': net.feature1.parameters(), 'lr': args.lr},
            {'params': net.feature2.parameters(), 'lr': args.lr},
            {'params': net.feature3.parameters(), 'lr': args.lr},
            {'params': net.feature4.parameters(), 'lr': args.lr},
            {'params': net.feature5.parameters(), 'lr': args.lr},
            {'params': net.feature6.parameters(), 'lr': args.lr},
            {'params': net.classifier1.parameters(), 'lr': args.lr},
            {'params': net.classifier2.parameters(), 'lr': args.lr},
            {'params': net.classifier3.parameters(), 'lr': args.lr},
            {'params': net.classifier4.parameters(), 'lr': args.lr},
            {'params': net.classifier5.parameters(), 'lr': args.lr},
            {'params': net.classifier6.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif optim_name == 'adam':
        optimizer = optim.Adam([
            {'params': base_params, 'lr': 0.1*args.lr},
            {'params': net.feature.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],weight_decay=5e-4)
    return optimizer

## setting different lr to baseline / classifiers
def set_ignored_params_rga(net):
    ignored_params = list(map(id, net.classifier1.parameters()))\
                    + list(map(id, net.classifier2.parameters()))\
                    + list(map(id, net.classifier3.parameters()))\
                    + list(map(id, net.classifier4.parameters()))\
                    + list(map(id, net.classifier5.parameters()))\
                    + list(map(id, net.classifier6.parameters()))
    return ignored_params

def set_optimizer_rga(optim_name,net,base_params):
    if optim_name == 'sgd':
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1*args.lr},
            {'params': net.classifier1.parameters(), 'lr': args.lr},
            {'params': net.classifier2.parameters(), 'lr': args.lr},
            {'params': net.classifier3.parameters(), 'lr': args.lr},
            {'params': net.classifier4.parameters(), 'lr': args.lr},
            {'params': net.classifier5.parameters(), 'lr': args.lr},
            {'params': net.classifier6.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif optim_name == 'adam':
        optimizer = optim.Adam([
            {'params': base_params, 'lr': 0.1*args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],weight_decay=5e-4)
    return optimizer


# *********** params for TSLFN *********** 
# ignored_params = set_ignored_params(net)
# base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
# optimizer = set_optimizer(args.optim, net, base_params)

# *********** params for TSLFN+RGA *********** 
ignored_params = set_ignored_params_rga(net)
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer = set_optimizer_rga(args.optim, net, base_params)


def adjust_learning_rate_rga(optimizer, epoch, change_epoch=[30,60]):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch < change_epoch[0]:
        lr = args.lr
    elif epoch >= change_epoch[0] and epoch < change_epoch[1]:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    
    optimizer.param_groups[0]['lr'] = 0.1*lr
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr
    optimizer.param_groups[3]['lr'] = lr
    optimizer.param_groups[4]['lr'] = lr
    optimizer.param_groups[5]['lr'] = lr
    optimizer.param_groups[6]['lr'] = lr
    return lr


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    if epoch < 30:
        lr = args.lr
    elif epoch >= 30 and epoch < 60:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    
    optimizer.param_groups[0]['lr'] = 0.1*lr
    optimizer.param_groups[1]['lr'] = lr
    optimizer.param_groups[2]['lr'] = lr
    optimizer.param_groups[3]['lr'] = lr
    optimizer.param_groups[4]['lr'] = lr
    optimizer.param_groups[5]['lr'] = lr
    optimizer.param_groups[6]['lr'] = lr
    optimizer.param_groups[7]['lr'] = lr
    optimizer.param_groups[8]['lr'] = lr
    optimizer.param_groups[9]['lr'] = lr
    optimizer.param_groups[10]['lr'] = lr
    optimizer.param_groups[11]['lr'] = lr
    optimizer.param_groups[12]['lr'] = lr
    return lr


def CalAccurate(cla_output,labels):
    _, predicted = cla_output.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct/len(labels)

def extract_feat(data_loader,data_num,forward_mode):
    net.eval()

    start = time.time()
    ptr = 0
    feats = np.zeros((data_num, 6*feature_dim))
    with torch.no_grad():
        for _, (input, _) in enumerate(data_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat = net(input, input, forward_mode)
            feats[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
            ptr += batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return feats 

def train(epoch):
    # current_lr = adjust_learning_rate(optimizer, epoch)
    current_lr = adjust_learning_rate_rga(optimizer, epoch, change_epoch=[40,80])
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0

    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        input1 = input1.cuda()
        input2 = input2.cuda()
        
        labels = torch.cat((label1,label2),0).cuda()
        label1 = label1.cuda()
        label2 = label2.cuda()

        data_time.update(time.time() - end)
        
        x_pool, outputs, feat = net(input1, input2) # outputs: classifier outputs -> id loss ; feat: feature after l2norm -> hc loss ; rga net
        # outputs, feat = net(input1, input2) # for original net
        
        # loss_tri, _ = criterion_tri(x_pool, labels)  
    
        if args.method =='id':
            loss0 = criterion(outputs[0], labels)
            loss1 = criterion(outputs[1], labels)
            loss2 = criterion(outputs[2], labels)
            loss3 = criterion(outputs[3], labels)
            loss4 = criterion(outputs[4], labels)
            loss5 = criterion(outputs[5], labels)
            
            het_feat0 = feat[0].chunk(2, 0)
            het_feat1 = feat[1].chunk(2, 0)
            het_feat2 = feat[2].chunk(2, 0)
            het_feat3 = feat[3].chunk(2, 0)
            het_feat4 = feat[4].chunk(2, 0)
            het_feat5 = feat[5].chunk(2, 0)

            loss_c0 = criterion_het(het_feat0[0], het_feat0[1], label1, label2)
            loss_c1 = criterion_het(het_feat1[0], het_feat1[1], label1, label2)
            loss_c2 = criterion_het(het_feat2[0], het_feat2[1], label1, label2)
            loss_c3 = criterion_het(het_feat3[0], het_feat3[1], label1, label2)
            loss_c4 = criterion_het(het_feat4[0], het_feat4[1], label1, label2)
            loss_c5 = criterion_het(het_feat5[0], het_feat5[1], label1, label2)
            loss0 = loss0 + w_hc * loss_c0
            loss1 = loss1 + w_hc * loss_c1
            loss2 = loss2 + w_hc * loss_c2
            loss3 = loss3 + w_hc * loss_c3
            loss4 = loss4 + w_hc * loss_c4
            loss5 = loss5 + w_hc * loss_c5

            cc = 0
            for ss in range(6):
                acc=eval("CalAccurate(outputs[{}],labels)".format(ss))
                cc += acc
            correct+=cc/6
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss_tri
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), 2*input1.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % loss_print_interval==0:
            print("=> ID loss {:.2f}".format(loss0-w_hc*loss_c0))
            print("=> HC loss {:.2f}".format(loss_c0))
            # print("=> Triplet loss {:.2f}".format(loss_tri))
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'lr:{:1.1e} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'Accu: {:.2f}%' .format(
                  epoch, batch_idx, len(trainloader), current_lr, 
                  100.*correct/len(trainloader), batch_time=batch_time, 
                  data_time=data_time, train_loss=train_loss))


def test(epoch):   
    gall_feat = extract_feat(gall_loader,ngall,test_mode[0])
    query_feat = extract_feat(query_loader,nquery,test_mode[1])

    start = time.time()
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    
    if dataset =='regdb':
        cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
    elif dataset =='sysu':
        cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time()-start))
    return cmc, mAP
    
# training
print('==> Start Training...')
per_img = args.per_img
per_id = args.batch_size / per_img
w_hc = args.w_hc

for epoch in range(start_epoch, args.epochs+1-start_epoch):
    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
        trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size, per_img)
    trainset.cIndex = sampler.index1 # color index
    trainset.tIndex = sampler.index2 # thermal index
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size,\
        sampler = sampler, num_workers=args.workers, drop_last =True)
    
    train(epoch)

    if epoch > 0 and epoch%2==0:
        print ('Test Epoch: {}'.format(epoch))
        print ('Test Epoch: {}'.format(epoch),file=test_log_file)

        cmc, mAP = test(epoch)

        print('FC:   Rank-1: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[9], cmc[19], mAP))
        print('FC:   Rank-1: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[9], cmc[19], mAP), file = test_log_file)
        test_log_file.flush()
        
        if mAP > best_mAP:
            best_mAP = mAP
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')


        # save model every 20 epochs    
        # if epoch > 10 and epoch%args.save_epoch ==0:
        #     state = {
        #         'net': net.state_dict(),
        #         'cmc': cmc,
        #         'mAP': mAP,
        #         'epoch': epoch,
        #     }
        #     torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))
