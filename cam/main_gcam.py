from __future__ import print_function

import copy
import os.path as osp

import cv2
import numpy as np
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from data_manager import *
from models.shared_model import embed_net_shared
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


def demo1(images, raw_images, model, id_label, target_layer, output_dir):
    bp = BackPropagation(model=model,forward_mode=forward_mode)
    _, ids = bp.forward(images)                                 # ids: pred id
    bp.remove_hook()
    
    gcam = GradCAM(model=model,forward_mode=forward_mode,target_layer=target_layer)
    _ = gcam.forward(images)
    
    gcam.backward(ids=ids[:, [0]])
    regions = gcam.generate(target_layer=target_layer)

    for j in range(len(images)):
        save_gradcam(filename=osp.join(output_dir,"{}-feat{}-gradcam-pre{}-gt{}.png".format(j,part_id,ids[j,0], id_label[j])),gcam=regions[j,0],raw_image=raw_images[j])


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


dataset = 'sysu'
output_dir = './vis/sysu_cam/'
data_path = './data/sysu/'
model_name = 'sysu_lr_1.0e-02_md_all_sharenet2_cla4_npart4_best.t'

low_dim = 512
n_class = 395 if dataset == 'sysu' else 206
img_h = 288
img_w = 144
share_net = 2
forward_mode = 2        # gallery=1; query=2 (sysu)
npart = 4
part_id = 0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),    
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
    normalize,
])

# Load model
model = embed_net_shared(low_dim, n_class, height=img_h, width=img_w, npart=npart, share_net=share_net, branch_name='rgas')
model = model.cuda()
model_path = 'save_model/' + dataset + '/' + model_name
model = load_model(model, model_path)
model.eval()

# Load data  
img_paths, id_label, _ = process_query_sysu(data_path, 'all')
# gall_img_path, gall_label, gall_cam = process_gallery_sysu(data_path, 'all')
n_img = len(img_paths)

sel_id = np.random.choice(n_img, size=10, replace=False)
img_paths = [img_paths[i] for i in sel_id]
id_label = np.array([id_label[i] for i in sel_id])
print("id label, ",id_label)
images,imgs_show = load_images(img_paths,transform_test)

target_layer = get_last_conv_name(model)

demo1(images, imgs_show, model, id_label, [target_layer], output_dir)