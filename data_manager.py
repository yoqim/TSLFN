from __future__ import print_function, absolute_import
import os
import sys
import numpy as np
import random

def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]
    
    files_ir = []
    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', relabel=False, gall_mode='single'):
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
    
    data_path = '../IVReIDData/SYSU-MM01/'
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]
    #np.random.seed(1)
    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                if gall_mode == 'single':
                    files_rgb.append(random.choice(new_files))
                if gall_mode == 'multi':
                    files_rgb.append(np.random.choice(new_files, 10, replace=False))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        if gall_mode == 'single':
            camid, pid = int(img_path[-15]), int(img_path[-13:-9])
            gall_img.append(img_path)
            gall_id.append(pid)
            gall_cam.append(camid)
        if gall_mode == 'multi':
            for i in img_path:
                camid, pid = int(i[-15]), int(i[-13:-9])
                gall_img.append(i)
                gall_id.append(pid)
                gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    
def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)