from __future__ import print_function

import copy
import os.path as osp

import cv2,sys
sys.path.append('..')
import numpy as np
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from data_manager import *
from models.shared_model import embed_net_shared,embed_net_mulcla
from models.model_ddag import embed_net_graph
from grad_cam import BackPropagation,GradCAM


def norm_image(image):
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def save_gradcam(filename, gcam, raw_image):
    gcam = gcam.cpu().numpy()
    gcam = cv2.resize(gcam, (raw_image.shape[1], raw_image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(raw_image)
    cam = norm_image(cam)
    cv2.imwrite(filename, cam)


def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def load_model(model,model_path):
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(model_name, checkpoint['epoch']))
        cur_mAP = checkpoint['mAP']
        print("cur mAP: {:.2f}%".format(cur_mAP*100))

    return model

def load_images(image_paths,transform):
    images = []
    imgs_show = []
    for ind, path in enumerate(image_paths):
        img_show = cv2.imread(path, 1)
        img_show = np.float32(img_show)/255
        imgs_show.append(img_show)

        img = Image.open(path)
        img = img.resize((img_w, img_h), Image.ANTIALIAS)
        img = np.array(img)
        img = transform(img)
        images.append(img)
    
    images = torch.stack(images).cuda()

    return images, imgs_show


def process_train_sysu(data_path):
    rgb_cameras = ['cam1','cam2','cam4','cam5']

    file_path = os.path.join(data_path,'exp/train_id.txt')
    file_path_val   = os.path.join(data_path,'exp/val_id.txt')

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_train = ["%04d" % x for x in ids]
    
    with open(file_path_val, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        id_val = ["%04d" % x for x in ids]

    id_train.extend(id_val)

    files_rgb = []
    for id in sorted(id_train):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)
    
    pid_container = set()
    for img_path in files_rgb:
        pid = int(img_path[-13:-9])
        pid_container.add(pid)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}

    img_paths = []
    pids = []
    for img_path in files_rgb:
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        pids.append(pid)

        img_paths.append(img_path)

    return img_paths, np.array(pids)


#################
# Main Functions
#################
def vis_cam(images, raw_images, model, id_label, target_layer, output_dir):
    # bp = BackPropagation(model=model,forward_mode=forward_mode)
    # sorted_ids = bp.forward(images) # ids: pred id
    # bp.remove_hook()

    gcam = GradCAM(model=model,forward_mode=forward_mode,target_layer=target_layer)
    sorted_ids = gcam.forward(images)

    id_label = torch.from_numpy(id_label).unsqueeze(1)
    id_label = id_label.cuda()

    for pid, sid in enumerate(sorted_ids):
        gcam.backward(ids=id_label, n_fea=pid)             # gt id backward
        # gcam.backward(ids=sid[:, [0]], n_fea=pid)            # pre id backward
    
    regions = gcam.generate_add()
    
    for j in range(len(images)):
        # if j in bad_id:
        save_gradcam(filename=osp.join(output_dir,"{}-gradcam_gt.png".format(j)), gcam=regions[j,0], raw_image=raw_images[j])


def get_bad_ids(images, model, id_label, target_layer):
    gcam = GradCAM(model=model,forward_mode=forward_mode,target_layer=target_layer)
    sorted_ids = gcam.forward(images)

    id_label = torch.from_numpy(id_label).unsqueeze(1)
    id_label = id_label.cuda()

    pred_label = torch.zeros((len(sel_id), npart)).cuda()
    for pid, sid in enumerate(sorted_ids):
        pred_label[:,pid] = sid[:, 0]
    
    t = pred_label.eq(id_label).sum(dim=1)
    bad_id = (t == 0).nonzero(as_tuple=True)[0] 
    bad_ids = sel_id[bad_id.cpu().numpy()]

    print("bad_id len, ",len(bad_id))
    print("bad ids, ",list(bad_ids))

    return bad_ids







dataset = 'sysu'
output_dir = '../vis/sysu_cam/'
data_path = '../data/sysu/'
model_name = 'sysu_lr_1.0e-02_md_all_sharenet3_mulcla4_graph_best.t'
model_path = '../save_model/' + dataset + '/' + model_name

low_dim = 512
n_class = 395 if dataset == 'sysu' else 206
img_h = 288
img_w = 144
share_net = 3
forward_mode = 2        # gallery=1 (RGB); query=2 (IR) (sysu)
npart = 4
sel_num = 64

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),    
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])

# Load model
model = embed_net_graph(low_dim, n_class, height=img_h, width=img_w, npart=npart, share_net=share_net, branch_name='rgas')
model = model.cuda()
model = load_model(model, model_path)
model.eval()

target_layer = get_last_conv_name(model)            # 'base_resnet.base.layer4.2.conv3'

# Load data  
# img_paths, id_label, _ = process_query_sysu('../../IVReIDData/SYSU-MM01/', 'all')
# img_paths, id_label, _ = process_gallery_sysu('../../IVReIDData/SYSU-MM01/', 'all')
img_paths, id_label = process_train_sysu('../../IVReIDData/SYSU-MM01/')
n_img = len(img_paths)


# sel_id = np.random.choice(n_img, size=sel_num, replace=False)
sel_id = [15767, 18692, 10895, 14711, 13251, 13261, 2459, 3436, 6127, 9111, 19554, 12742, 14710, 9000, 21234, 5135, 17256, 4950, 18598, 2069, 6959]

img_paths = [img_paths[i] for i in sel_id]
id_label = np.array([id_label[i] for i in sel_id])
images,imgs_show = load_images(img_paths,transform_test)

print("=> get_bad_ids")
bad_ids = get_bad_ids(images, model, id_label, [target_layer])


print("=> vis_cam")
vis_cam(images, imgs_show, model, id_label, [target_layer], output_dir)