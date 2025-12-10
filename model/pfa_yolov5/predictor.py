from models.common import DetectMultiBackend
import torch
from model.pfa_yolov5.utils.general import (
    scale_boxes,
    xyxy2xywh,
    check_img_size,
    non_max_suppression,
)
from model.pfa_yolov5.utils.augmentations import letterbox
import numpy as np
from model.pfa_yolov5.utils.plots import Annotator
import random
from typing import List
import cv2


class YOLOv5Detector:

    def __init__(
        self,
        weights_path,
        img_size=(640, 640),
        conf_thres=0.15,
        iou_thres=0.30,
        max_det=10,
        device="cuda",
        classes_name: List[str] =  ['B1','B2','B3','B4','B5','B7',
        'R1','R2','R3','R4','R5','R7'],
        classes=None,
        agnostic_nms=False,
        augment=False,
        half=True,
        visualize=True,
    ):
        self.device = torch.device(device)
        self.model = DetectMultiBackend(
            weights=weights_path, device=self.device, dnn=False, fp16=True, fuse=True
        )

        stride, names, pt, jit, onnx, engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        self.names = classes_name
        self.img_size = check_img_size(img_size, s=stride)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.half = half and (pt or jit or onnx or engine) and self.device.type != "cpu"

        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.augment = augment
        self.visualize = visualize
        self.agnostic_nms = agnostic_nms

    def preprocess(self, img):
        im = letterbox(img, self.img_size, self.model.stride, auto=self.model.pt)[0]
        # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW
        im = im.transpose((2, 0, 1))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def preprocess_batch(self, img: List[np.ndarray]):
        # 强制禁用 auto=True，确保所有图片都被 resize 到完全相同的尺寸 (self.img_size)
        # 否则 letterbox 会根据长宽比自动调整 stride，导致不同图片尺寸不一致，无法 stack
        im = [
            letterbox(image, self.img_size, self.model.stride, auto=False)[0]
            for image in img
        ]
        # im = [image.transpose((2, 0, 1))[::-1] for image in im]
        im = [image.transpose((2, 0, 1)) for image in im]
        im = [np.ascontiguousarray(image) for image in im]
        im = [torch.from_numpy(image).to(self.device) for image in im]
        im = [image.half() if self.half else image.float() for image in im]
        im = [image / 255 for image in im]
        ims = (
            torch.stack(im) if len(im) > 1 else im[0].unsqueeze(0)
        )  # stack if batch size > 1
        return ims

    def inference(self, img):
        if isinstance(img, list):  # List of images
            im = self.preprocess_batch(img)
        else:
            im = self.preprocess(img)
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        return pred

    def postprocess(self, pred, im, im0):
        pred = non_max_suppression(
            prediction=pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
        )

        detections = []
        annot_img = None

        for i, det in enumerate(pred):
            if len(det):
                # 确保 im0 是单张图片，而不是 batch 列表
                current_im0 = im0[i] if isinstance(im0, list) else im0
                
                # scale_boxes 需要正确的原始图片尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], current_im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # 修复：直接使用 xyxy，不需要转换成 xywh 再转回来
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    
                    if self.visualize:
                        annotator = Annotator(
                            np.ascontiguousarray(current_im0),
                            line_width=3,
                            example=str(self.names),
                        )
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])

                    line = (int(cls), list(map(lambda x: float(x), xyxy)), float(conf))
                    detections.append(line)

                if self.visualize:
                    annot_img = annotator.result()

        return detections, annot_img

    def predict(self, img):
        # im0 = img.copy()
        im0 = img
        im = self.preprocess(img)
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        detections = self.postprocess(pred, im, im0)
        return detections

    def predict_batch(self, img: List[np.ndarray]) -> List[List[tuple]]:
        """
        Predict detections for a batch of images.

        Args:
            img: List of input images (NumPy arrays).

        Returns:
            List of detections for each image, where each detection is a tuple
            (class_name, [x, y, w, h], score).
        """
        im0 = [image.copy() for image in img]
        im = self.preprocess_batch(img)
        pred = self.model(im, augment=self.augment, visualize=self.visualize)

        # 如果 pred 是 list 或 tuple，通常第一个元素是推理结果 Tensor (Batch, Num_Anchors, 85)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        results = []
        annot_imgs = []
        for i in range(len(img)):
            # pred 现在应该是 Tensor (Batch, Num_Anchors, 85)
            # 取第 i 个样本的预测结果
            current_pred = pred[i]

            # 确保维度正确: (Num_Anchors, 85) -> (1, Num_Anchors, 85) 以便传入 postprocess
            if len(current_pred.shape) == 2: 
                 current_pred = current_pred.unsqueeze(0)

            # 处理 im: im 是 Tensor (Batch, C, H, W)，切片后是 (C, H, W)，需要 unsqueeze 变回 (1, C, H, W)
            current_im = im[i].unsqueeze(0)

            detections, annot_img = self.postprocess(
                current_pred, current_im, im0[i]
            )
            results.append(detections)
            if self.visualize:
                annot_imgs.append(annot_img)

        return results, annot_imgs


if __name__ == "__main__":
    predictor = YOLOv5Detector(
        weights_path="weights/car_pfa.pt",
        visualize=True,
        img_size=(640, 640),
    )