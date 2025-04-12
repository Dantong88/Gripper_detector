import argparse
import argparse
import logging
import os
from collections import OrderedDict
import joblib
import torch
import torch.nn as nn
import random
import pdb
import cv2
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from torchvision import transforms
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

class BatchPredictor(nn.Module):
    """
    The batch version of detectron2 DefaultPredictor
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def forward(self, imgs):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            inputs = []
            for original_image in imgs:
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                entry = {"image": image, "height": height, "width": width}
                inputs.append(entry)

            # inference
            predictions = self.model(inputs)

            return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference of end-effector detector')
    parser.add_argument('--config_file', type=str,
                        default='config/3dprint_config.yaml')
    # parser.add_argument('--ckpt', type=str,
    #                     default='/home/niudt/robotic_project/detectron2/tools/lego_40epoch/model_final.pth')
    parser.add_argument('--ckpt', type=str,
                        default='/home/niudt/robotic_project/detectron2/tools/3d_print_40epoch/model_final.pth')
    parser.add_argument('--device', type=str,
                        default='cpu')
    parser.add_argument('--input_img', type=str,
                        default='/scratch/partial_datasets/lvma/lego_project/data/pkl_adapted/2025-02-09/peg_2-4-6/02-09-2025/2025-02-09T18:59:05.944334/first_frames/rgb_left.png')
    parser.add_argument('--save_path', type=str,
                        default='assets')
    parser.add_argument('--conf_thresh', type=float,
                        default=0.5)

    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.ckpt  # add model weight here
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_thresh  # 0.5 , set the testing threshold for this model
    predictor = BatchPredictor(cfg).to(args.device)

    # this is for the lego blocks
    supported_objs = [{'id': 0, 'name': 'blue cube', 'color': [220, 20, 60], 'isthing': 1}]

    # here is the forloop to process the data
    # demos = glob(os.path.join('/scratch/partial_datasets/general_manipulation/data/pkl_crop/150_pinkcubes_2025-03-24_Eric', '*'))
    demos = glob(
        os.path.join('/datasets/lvma/current/lego/data/franka/pkl/*', '*'))
    # demos = ['/scratch/partial_datasets/lvma/lego_project/data/pkl_adapted/ball/16/02-09-2025/2025-02-09T21:55:34.849549']
    # get detection box
    for demo in tqdm(demos):
        bbox_det = {}
        for view in ['rgb_left.png', 'rgb_right.png']:
            input_img = os.path.join(demo, 'first_frames', view)
            image_cv2 = cv2.imread(input_img)
            visualizer = Visualizer(img_rgb=image_cv2[:, :, ::-1])
            image_cv2 = cv2.imread(input_img) # Convert from RGB to BGR to align with the image format in OpenCV
            det_pred = predictor([image_cv2])

            if len(det_pred) != 0:
                bbox = det_pred[0]['instances'].pred_boxes.tensor.detach().numpy()[0]
                bbox = [round(i) for i in bbox]
                bbox_det[view] = bbox


            else:
                print('no detected boxes in ', demo)


        # write the pkl
        pkls = glob(os.path.join(demo, '*'))

        for pkl in pkls:
            if not pkl.endswith('.pkl'):  # More robust check for '.pkl' files
                continue
            with open(pkl, "rb") as f:
                data = joblib.load(f)  # Load the dictionary

            for key in list(data.keys()):
                if 'bbox' in key:
                    del data[key]

            # Ensure bbox_det has the required keys before updating `data`
            if 'rgb_left.png' in bbox_det and 'rgb_right.png' in bbox_det:
                data['bbox_left'] = bbox_det['rgb_left.png']
                data['bbox_right'] = bbox_det['rgb_right.png']

                # Save the updated dictionary
                with open(pkl, "wb") as f:
                    joblib.dump(data, f)
            else:
                print(f"Warning: Missing keys in bbox_det for file {pkl}")



