#coding:UTF-8
#!/usr/bin/env python2

import os
os.environ["GLOG_minloglevel"] = '5'
os.environ['TFU_ENABLE']='1' 
os.environ['TFU_NET_FILTER']='0' 
os.environ['CNRT_PRINT_INFO']='false' 
os.environ['CNRT_GET_HARDWARE_TIME']='false'
os.environ['CNML_PRINT_INFO']='false'  
import caffe
import math
import shutil
import stat
import subprocess
import sys
import numpy as np
import collections
import copy
import time
import traceback
import datetime
import cv2

import matplotlib.pyplot as plt
from PIL import Image
plt.switch_backend('agg')
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

#np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=np.inf)

def get_boxes(prediction, batch_size, img_size=416):
    """
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    reshape_value = prediction.reshape((-1, 1))
    num_boxes_final = reshape_value[0].item()
    print(num_boxes_final)
    all_list = [[] for _ in range(batch_size)]
    max_limit = 1
    min_limit = 0
    for i in range(int(num_boxes_final)):
        batch_idx = int(reshape_value[64 + i * 7 + 0].item())
        if batch_idx >= 0 and batch_idx < batch_size:
            bl = max(min_limit, min(max_limit, reshape_value[64 + i * 7 + 3].item()) * img_size)
            br = max(min_limit, min(max_limit, reshape_value[64 + i * 7 + 4].item()) * img_size)
            bt = max(min_limit, min(max_limit, reshape_value[64 + i * 7 + 5].item()) * img_size)
            bb = max(min_limit, min(max_limit, reshape_value[64 + i * 7 + 6].item()) * img_size)

            if bt - bl > 0 and bb -br > 0:
                all_list[batch_idx].append(bl)
                all_list[batch_idx].append(br)
                all_list[batch_idx].append(bt)
                all_list[batch_idx].append(bb)
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 2].item())
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 2].item())
                all_list[batch_idx].append(reshape_value[64 + i * 7 + 1].item())

    output = [np.array(all_list[i]).reshape(-1, 7) for i in range(batch_size)]
    return output
    
def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    fp.close()

    return names

if __name__ == '__main__':
    if len(sys.argv)!=3:
        print("Usage:{} prototxt caffemodel".format(sys.argv[0]))
        sys.exit(1)
    
    batch_size=1
    prototxt=sys.argv[1]
    caffemodel=sys.argv[2]
    
    caffe.set_mode_mfus()
    #caffe.set_mode_mlu()
    caffe.set_core_number(1)
    caffe.set_batch_size(batch_size)
    caffe.set_simple_flag(1)
    
    caffe.set_rt_core("MLU270")
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    input_name = net.blobs.keys()[0]

    label_path = './label_map_coco.txt'
    classes = load_classes(label_path)
    
    image_path = './images/image.jpg'
    img=cv2.imread(image_path)
    img = img[:, :, (2, 1, 0)]
    
    h, w, _ = img.shape
    scale = float(416)/h if w < h else float(416)/w
    # get new w and h
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
    dim_diff = np.abs(new_h - new_w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    input_img = np.pad(img, pad, 'constant', constant_values=128)
    
    image = np.transpose(input_img, (2, 0, 1))
    image=image[np.newaxis, :].astype(np.float32)
    images = np.repeat(image, batch_size, axis=0)
    
    net.blobs[input_name].data[...]=images 

    output = net.forward()
    output_keys=output.keys()
    output=output[output_keys[0]].astype(np.float32)
    outputs = get_boxes(output, batch_size, img_size=416)
    #print(outputs)
    out_img = np.array(Image.open(image_path))
    for si, pred in enumerate(outputs):
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(out_img)
        # The amount of padding that was added        
        pad_x = max(out_img.shape[0] - out_img.shape[1], 0) * ( float(416) / max(out_img.shape))
        pad_y = max(out_img.shape[1] - out_img.shape[0], 0) * ( float(416) / max(out_img.shape))
        # Image height and width after padding is removed
        unpad_h = 416 - pad_y
        unpad_w = 416 - pad_x
        if pred is not None:
            file = open('output/out_img.txt', mode='w')
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in pred:
                print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * out_img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * out_img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * out_img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * out_img.shape[1]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                        edgecolor='red',
                                        facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label and confidence
                con = format(cls_conf.item(), '.3f')
                s = classes[int(cls_pred)] + ':' + str(con)
                plt.text(x1, y1, s, color='blue', verticalalignment='top')
                # Save the result of detections 
                file.write('\t+ Label: %s, Conf: %.5f\n' % (classes[int(cls_pred)], cls_conf.item()))
            file.close()
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('output/out_img.png', bbox_inches='tight', pad_inches=0.0)
        plt.close()

