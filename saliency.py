
import cv2
import numpy as np

from data_manager import *


save_dir = './vis/sal_sysu/'
data_path = './data/sysu/'

gall_img_path, gall_label, gall_cam = process_gallery_sysu(data_path, 'all')
n_img = len(gall_img_path)

sel_id = np.random.choice(a=n_img, size=10, replace=False)
gall_img_path = [gall_img_path[i] for i in sel_id]

for gi, path in enumerate(gall_img_path):
    image = cv2.imread(path)

    # saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    print(np.max(saliencyMap))

    threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    concat_pic = np.concatenate([saliencyMap, threshMap], axis=1)
    cv2.imwrite(save_dir + "Image_{}.png".format(gi),image)
    cv2.imwrite(save_dir + "Output_{}.png".format(gi),concat_pic)

    if gi%10==0 and gi>0:
        import pdb;pdb.set_trace()


