import os
import json
import glob 
import math 
import json
from collections import OrderedDict
import base64
import random
import numpy as np
import cv2
import detectron2
import torch, torchvision
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

setup_logger()

class Detectron_inference():
    def __init__(self, GPU_NUM = '1'):
        self.set_gpu(GPU_NUM)
        self.class_names = ['cell']
        pth_path = "/home/ksShin/all_defect_classification/detectron2/Detectron2-Train-a-Instance-Segmentation-Model/output/model_final.pth"
        self.set_cfp(pth_path = pth_path)
        
    def set_gpu(self, GPU_NUM):
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device) # change allocation of current GPU
        print ('Current cuda device ', torch.cuda.current_device()) # check

    def set_cfp(self, pth_path):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        self.cfg.MODEL.WEIGHTS = pth_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
        self.predictor = DefaultPredictor(self.cfg)
        
    def empty_gpu(self):
        #del self.cfg
        #del self.predictor
        torch.cuda.empty_cache()

    def get_result(self, img_path):
        self.img = cv2.imread(img_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.outputs = self.predictor(self.img)
    
    def display_result(self):
        v = Visualizer(self.img[:, :, ::-1],
                       scale=1, 
                       instance_mode=ColorMode.IMAGE_BW)
        v = v.draw_instance_predictions(self.outputs["instances"].to("cpu"))
        self.result_img = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        return self.result_img
    
    def filter_out_image(self):
        ins = self.outputs["instances"].get_fields()
        cnt_obj = len(ins['pred_boxes'])
        base_mask = np.full(self.outputs['instances'].image_size, 0, dtype='uint8')
        
        for i in range(cnt_obj):
            pred_masks = np.array(ins['pred_masks'][i].cpu(), dtype='uint8')
            base_mask = cv2.bitwise_or(base_mask,pred_masks)
        base_mask = cv2.bitwise_not(base_mask)
        base_mask = (base_mask / 255).astype('uint8')
        reverse_img = cv2.bitwise_and(self.img,self.img, mask = base_mask)        
        return reverse_img
    
    def result_info(self):
        img_area = self.outputs['instances'].image_size[0] * self.outputs['instances'].image_size[1]
        area_list = []
        area_per_list = []
        obj_list = []
        result_info = {}
        for i in range(len(self.outputs['instances'])):
            obj_mask = np.array(self.outputs['instances'][i].get_fields()['pred_masks'].cpu())
            obj_area = round(np.count_nonzero(obj_mask) / img_area,3)
            obj_list.append(i)
            area_per_list.append(obj_area)
            area_list.append(np.count_nonzero(obj_mask))
        
#        result_info = {"obj_num":obj_list,
#                        "obj_size": area_list,
#                       "obj_size_per":area_per_list}
#        result_info = json.dumps(result_info)
        return obj_list, area_list, area_per_list
        
    def Main(self, image_path):
        self.get_result(image_path)
        result_img = self.display_result()
        result_info = self.result_info()
        filter_out_img = self.filter_out_image()
        cv2.imwrite('./test.jpg', result_img)
        cv2.imwrite('./test2.jpg', filter_out_img)
        return result_info
        
if __name__ == '__main__':
    detectron = Detectron_inference()
    detectron.Main(image_path = './tmp/CFP 1-3.jpg')