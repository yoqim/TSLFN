import os,sys,cv2
import numpy as np


res_dir = './vis/resnet/'
rga_dir = './vis/rga/'

res_path = [res_dir + i for i in os.listdir(res_dir) if i.endswith('.jpg') and i.startswith('regdb')]
# rga_path = [rga_dir + i for i in os.listdir(rga_dir) if i.endswith('.jpg')]

for resp in res_path:
    person_id = resp.split('/')[-1].split('_')[-2]
    feat_id = resp.split('/')[-1].split('_')[-1].strip('.jpg')

    rgap = rga_dir + 'regdb_rga_feamap_' + person_id+ '_'+ feat_id + '.jpg'
    if not os.path.exists(rgap):
        print(rgap, " doesn't exists!")
        continue
    print(resp)
    print(rgap)
    res_img= cv2.imread(resp)
    rga_img= cv2.imread(rgap)
    
    htitch= np.hstack((res_img, rga_img))
    del res_img,rga_img

    cv2.imwrite("./vis/comp/featmap_{}_{}.jpg".format(person_id,feat_id),htitch)
    cv2.destroyAllWindows()

    
