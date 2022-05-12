from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import Dataset
from torchvision.models.feature_extraction import create_feature_extractor


class RandomShapes:

    def __init__(self, img_size: int, w: int, h: int, seed: int, nms_threshold: float = 0.3,
                 bottom_right: bool = False, background_class: bool = True):
        """
        Original shapes dataset: https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/shapes.py
        :param img_size:         image size
        :param w:                object width
        :param h:                object height
        :param bottom_right:        return (x2,y2) as bottom_right instead of w,h
        :param background_class: use background class 0
        :param seed:             numpy random seed
        :param nms_threshold: nms to select the degree of figures intersection for each image
        """
        self.seed = seed
        self.img_size = img_size
        self.w = w
        self.h = h
        self.bottom_right = bottom_right
        self.background_class = background_class
        self.nms_threshold = nms_threshold
        self.classes = ('circle', 'triangle', 'square')
        np.random.seed(self.seed)

    def get_class_id(self, name: str) -> int:
        """
        +1 means that background class is 0
        :param name: class name
        :return: class id
        """
        class_id = self.classes.index(name)
        class_id += 1 if self.background_class else 0
        return class_id

    def ground_truth(self, class_name: str, dims: list or tuple) -> Tuple[int, np.ndarray]:
        """
        Return GT class_id
        :param class_name: class name
        :param dims:       dims: center x, y and the size s
        :return: bbox format [x, y, w, h]
        """
        x, y, s = dims
        class_id = self.get_class_id(class_name)
        if self.bottom_right:
            bbox = np.array([x, y, x + s, y + s])
        else:
            bbox = np.array([max(x - s, 0), max(y - s, 0), 2 * s, 2 * s])
        return class_id, bbox

    def random_shape(self, height, width) -> tuple:
        """
        Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        :param height:
        :param width:
        :return:
        """
        # Shape
        shape = np.random.choice(self.classes)
        # Color
        color = np.random.randint(0, 255, 3)
        y = np.random.randint(0, self.img_size)
        x = np.random.randint(0, self.img_size)
        # Size
        s = np.random.randint(height // 4, height // 2)
        return shape, color, (x, y, s)

    def random_spec(self) -> tuple:
        """
        Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        :return:
        """
        # Pick random background color
        bg_color = np.random.randint(0, 255, 3)
        # Generate a few random shapes and record their bounding boxes
        shapes = []
        boxes = []
        N = np.random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(self.h, self.w)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([x - s, y - s, x + s, y + s])
        # Apply non-max suppression with 0.3 threshold to avoid shapes covering each other
        if N > 1:
            keep_ixs = torchvision.ops.nms(torch.Tensor(np.array(boxes)),
                                           torch.Tensor(np.arange(N)),
                                           self.nms_threshold)
            shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

    def draw_shape(self, image, shape, dims, color) -> np.ndarray:
        """
        Draws a shape from the given specs
        :param image:
        :param shape:
        :param dims:
        :param color:
        :return:
        """
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            image = cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color.tolist(), -1)
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color.tolist(), -1)
        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / np.math.sin(np.math.radians(60)), y + s),
                                (x + s / np.math.sin(np.math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color.tolist())
        return image

    def get_image(self) -> Tuple[np.ndarray, list]:
        bg_color, shape_specs, gtruth = self.generate_shapes()
        image = self.generate_image_by_desc(bg_color, shape_specs)
        return image, gtruth

    def generate_shapes(self) -> Tuple[np.ndarray, tuple, list]:
        bg_color, shape_specs = self.random_spec()
        gtruth = []
        for shape, color, dims in shape_specs:
            gtruth.append(self.ground_truth(shape, dims))
        return bg_color, shape_specs, gtruth

    def generate_image_by_desc(self, bg_color: np.ndarray, shape_specs: tuple) -> np.ndarray:
        """
        :param bg_color:
        :param shape_specs:
        :return:
        """
        image = np.ones([self.img_size, self.img_size, 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in shape_specs:
            image = self.draw_shape(image, shape, dims, color)
        return image


def prepare_proposals_dataset(data: Dataset) -> Tuple[torch.Tensor, np.array]:
    """
    Prepare proposals dataset for R-CNN classification and regression models
    :param data:
    :return:
    """
    images, classes = [], []
    loader = iter(data)
    svm_classes = data.svm_proposals_classes

    for batch, svm_class in zip(loader, svm_classes):
        crop, _ = batch
        images.append(crop.unsqueeze(0))
        classes.append(svm_class)

    images = torch.cat(images, 0)
    classes = np.array(classes)
    return images, classes


def get_proposals_features(model, batches: tuple, key: str, device: str, pbar: bool) -> np.ndarray:
    """
    Get proposal features from images
    :param model:
    :param batches:
    :param key:
    :param device:
    :param pbar:
    :return:
    """
    features = []
    iterable = tqdm.tqdm(batches, leave=True) if pbar else batches
    for b in iterable:
        b = b.to(device)
        with torch.no_grad():
            output = model.forward(b)[key].cpu().numpy()
            features.append(output)
    return np.concatenate(features)


def get_svm_dataset(model: nn.Module, dataset: Dataset, device: str, batch: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Prepare SVM classifier tabular data:
    # X - features from CNN
    # y - class ids
    Data with -1 label (0.3<=IoU<0.5) will be dropped.
    :param model:
    :param dataset:
    :param device:
    :param batch:
    :return:
    """

    images, classes = prepare_proposals_dataset(dataset)
    extractor = create_feature_extractor(model, {'flatten': 'features'})
    extractor = extractor.eval()
    X = get_proposals_features(model=extractor, batches=torch.split(images, batch),
                               key='features', device=device, pbar=True)
    y = classes
    X = X[y != -1]
    y = y[y != -1]
    X = X.reshape(X.shape[0], -1)
    return X, y


def get_bboxes_dataset(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Bbox regression dataset:
    # bbox_proposal - [x, y, w, h] from selective search
    # bbox_gt -       [x, y, w, h] from ground truth
    :param dataset:
    :return:
    """

    bbox_proposal = dataset.regression_bbox_pairs[:, 0, :]
    bbox_gt = dataset.regression_bbox_pairs[:, 1, :]
    return bbox_proposal, bbox_gt


def bbox2center(bbox_array: np.ndarray) -> np.ndarray:
    """
    Convert [x,y,w,h] bbox format into [X_center, Y_center, Size_w, Size_h] format
    :param bbox_array: np.ndarray [N, 4]
    :return: np.ndarray
    """
    bbox_array[:, 2] = bbox_array[:, 2] // 2
    bbox_array[:, 3] = bbox_array[:, 3] // 2
    bbox_array[:, 0] = bbox_array[:, 0] + bbox_array[:, 2]
    bbox_array[:, 1] = bbox_array[:, 1] + bbox_array[:, 3]

    return bbox_array


def center2bbox(bbox_array: np.ndarray) -> np.ndarray:
    """
    Convert [X_center, Y_center, Size_w, Size_h] bbox format into [x,y,w,h] format
    :param bbox_array:
    :return:
    """
    bbox_array[:, 0] = bbox_array[:, 0] - bbox_array[:, 2]
    bbox_array[:, 1] = bbox_array[:, 1] - bbox_array[:, 3]
    bbox_array[:, 2] = bbox_array[:, 2] * 2
    bbox_array[:, 3] = bbox_array[:, 3] * 2
    return bbox_array


def get_regression_targets(proposals: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Prepare regression targets for R-CNN:
    tx = (Gx - Px) / Px
    ty = (Gy - Py) / Ph
    tw = ln(Gw/Pw)
    th=  ln(Gh/Ph)
    :param proposals: P = (Px, Py, Pw, Ph)
    :param gt:        G = (Gx, Gy, Gw, Gh)
    :return: tx, ty, tw, th
    """

    proposals = proposals.astype(float)
    gt = gt.astype(float)

    tx = (gt[:, 0] - proposals[:, 0]) / proposals[:, 2]
    ty = (gt[:, 1] - proposals[:, 1]) / proposals[:, 3]
    tw = np.log(gt[:, 2] / proposals[:, 2])
    th = np.log(gt[:, 3] / proposals[:, 3])
    t = np.hstack([tx.reshape(-1, 1), ty.reshape(-1, 1), tw.reshape(-1, 1), th.reshape(-1, 1)])
    return t


def get_regression_dataset(model: nn.Module, dataset: Dataset, feature_layer: str,
                           device: str, batch: int = 150, pbar: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Prepare tabular data for bbox regressions
    X - features from CNN
    targets - [tx, ty, tw, th]
    Data with -1 and 0 labels (0.3<=IoU<0.5) will be dropped.
    :param model:
    :param dataset:
    :param device:
    :param batch:
    :param feature_layer:
    :param pbar:
    :return:
    """

    images, classes = prepare_proposals_dataset(dataset)
    extractor = create_feature_extractor(model, {feature_layer: 'features'})
    extractor = extractor.eval()
    X = get_proposals_features(model=extractor, batches=torch.split(images, batch),
                               key='features', device=device, pbar=pbar)

    bbox_proposal, bbox_gt = get_bboxes_dataset(dataset)
    bbox_proposal = bbox2center(bbox_proposal.copy())
    bbox_gt = bbox2center(bbox_gt.copy())
    targets = get_regression_targets(bbox_proposal, bbox_gt)

    X = X[(classes != 0) & (classes != -1)]
    X = X.reshape(X.shape[0], -1)
    y = targets[(classes != 0) & (classes != -1)]

    return X, y


def crop_image(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox
    crop = image[y:y + h, x:x + w]
    return crop


def plot_confusion_matrix(cmatrix: np.ndarray, class_names: list, title: str) -> None:
    """
    sklearn confusion matrix plotter
    :param cmatrix:
    :param class_names:
    :param title:
    :return:
    """
    disp = ConfusionMatrixDisplay(cmatrix)
    disp.display_labels = class_names
    disp.plot()
    plt.title(title)
    plt.show()


def plot_scores(layer_names: list, train_path: str, val_path: str, title: str) -> None:
    """
    Plot scores for R-CNN bbox regressions
    :param layer_names:
    :param train_path:
    :param val_path:
    :param title:
    :return:
    """
    train_scores = np.load(train_path)
    val_scores = np.load(val_path)
    best_layer_idx = np.argmin(val_scores)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(val_scores, label='val')
    plt.plot(train_scores, label='train')
    plt.scatter(np.argmin(val_scores), min(val_scores), color='green', label=f'lowest RMSE: %.3f' % min(val_scores))
    plt.scatter(np.argmax(val_scores), max(val_scores), color='red', label=f'highest RMSE: %.3f' % max(val_scores))

    plt.ylabel('RMSE')
    plt.xlabel('Layer number')
    plt.legend()
    plt.title(f'{title}. Best layer: {layer_names[best_layer_idx]}')
    plt.show()
