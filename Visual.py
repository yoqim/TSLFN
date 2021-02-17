import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms

from data_manager import *

data_path = '../IVReIDData/RegDB/'

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = alpha*heatmap + (1-alpha)*np.float32(img)
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)

alpha = 0.5
query_feat = np.load('./feat/regdb_query_feat_rga.npy')
query_img_path, query_label = process_test_regdb(data_path, trial=1, modal='visible')

for i,qp in enumerate(query_img_path):
    if i%20!=0:
        continue
    img = cv2.imread(qp)
    img = np.float32(img) / 255

    qf = query_feat[i,:]
    print("=======")
    for q in range(qf.shape[0]):
        if q%300!=0:
            continue
        print(q)

        cqf = qf[q,::]
        rqf = cv2.resize(cqf, (img.shape[1], img.shape[0]))
        img_add = show_cam_on_image(img, rqf)
        cv2.imwrite("vis/regdb_rgas_feamap_{}_{}.jpg".format(i,q), img_add)

    import pdb;pdb.set_trace()

