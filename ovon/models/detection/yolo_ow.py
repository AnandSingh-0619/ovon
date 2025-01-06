import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import torch
import cv2
import numpy as np
from PIL import Image
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from ultralytics import YOLO
from habitat.core.logging import logger
from gym import spaces
from ovon.models.detection.constants import CLASSES
from ovon.models.encoders.mask_encoder import mask_encoder
import wandb
class YOLOPerception():
    def __init__(
        self,
        yolo_model_id: Optional[str] = "yolov8x-worldv2.pt",
        verbose: Optional[bool] = False,
        confidence_threshold: Optional[float] = 0.002,
    ):
        """Loads a YOLO model for object detection and instance segmentation

        Arguments:
            yolo_model_id: yolo model to be used for detection
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """

        vocab = CLASSES
        with torch.no_grad():
            self.model = YOLO(model=yolo_model_id)
            self.model.set_classes(vocab)
        # Freeze the YOLO model's parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.confidence_threshold = confidence_threshold
        self.model.cuda()
        self.mask_feats = mask_encoder(mask_encoding_method="cnn")
        self.image_counter = 0
        if verbose:
            logger.info(f"Loading YOLO model: {yolo_model_id} ")
            total_yolo_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Total number of parameters in YOLO model: {total_yolo_params}"
            )
        self.output_size = 768
        self.output_shape = (self.output_size,)

    def predict(
        self,
        obs: spaces.Dict
    ) :
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - Detic expects BGR)

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with

        """

        nms_threshold = 0.8

        images_tensor = obs["rgb"]
        obj_class_ids = obs["yolo_object_sensor"].cpu().numpy().flatten()

        batch_size = images_tensor.shape[0]
        images = [images_tensor[i].cpu().numpy() for i in range(images_tensor.size(0))]

        height, width, _ = images[0].shape
        results = list(
            self.model(
                images,
                classes=obj_class_ids,
                conf=self.confidence_threshold,
                iou=nms_threshold,
                stream=True,
                verbose=False,
            )
        )
        total_detections = 0
        semantic_masks = []
        for idx, result in enumerate(results):
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            input_boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            # image = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
            semantic_mask = np.zeros((height, width, 1))

            if class_ids.size != 0:
                obj_mask_idx = np.isin(class_ids, obj_class_ids[idx])

                obj_boxes = input_boxes[obj_mask_idx]
                obj_confidences = confidences[obj_mask_idx]
                if len(obj_boxes) > 0:
                    total_detections += 1

                    # Select the box with the highest confidence
                    max_conf_idx = np.argmax(obj_confidences)
                    best_box = obj_boxes[max_conf_idx]

                    # Create a binary mask for the selected box
                    x_min, y_min, x_max, y_max = best_box.astype(int)
                    semantic_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

            # semantic_mask = cv2.resize(
            #     semantic_mask, (120, 160), interpolation=cv2.INTER_NEAREST
            # )
            # semantic_mask = np.expand_dims(semantic_mask, axis=-1)
            semantic_masks.append(semantic_mask)



        torch.cuda.empty_cache()
        semantic_masks = np.array(semantic_masks)
        semantic_masks = (
            torch.tensor(semantic_masks,dtype=torch.float32,device=torch.device("cuda:{}".format(torch.cuda.current_device())),).detach().requires_grad_(False)
        )
        mask_feats = self.mask_feats(semantic_masks)
        
        if wandb.run is not None:
            wandb.log({"total_detections": total_detections})
        # logger.info(f"Tracked total_detections: {total_detections}")
        return mask_feats 
