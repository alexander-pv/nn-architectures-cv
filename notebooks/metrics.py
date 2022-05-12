import numpy as np


def get_iou(bbox_a: tuple or np.ndarray, bbox_b: tuple or np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute IoU
    :param bbox_a: tuple of ints, (x1, y1, w1, h1)
    :param bbox_b: tuple of ints, (x2, y2, w2, h2)
    :param eps: float
    :return: float, IoU
    """

    # Rectangles a, b. Top left and bottom right dots
    x1_a, y1_a, w_a, h_a = bbox_a
    x1_b, y1_b, w_b, h_b = bbox_b

    x2_a = x1_a + w_a
    y2_a = y1_a + h_a
    x2_b = x1_b + w_b
    y2_b = y1_b + h_b

    # Intersection coordinates
    x_left = max(x1_a, x1_b)
    x_right = min(x2_a, x2_b)
    y_top = max(y1_a, y1_b)
    y_bottom = min(y2_a, y2_b)

    if x_left >= x_right or y_top >= y_bottom:
        intersect = 0.
    else:
        intersect = (x_right - x_left) * (y_bottom - y_top)

    # Check intersection existence
    if max(0, intersect):
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - intersect
        iou = intersect / (union + eps)
        assert iou >= 0.
        assert iou <= 1.
    else:
        iou = 0.

    return iou
