import os
from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.handlers import tensorboard_logger as tbl
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Loss, RunningAverage
from torch.autograd import Variable
from torch.utils.data import DataLoader


def train_model(train_loader: DataLoader,
                val_loader: DataLoader,
                dataset_name: str,
                model: nn.Module,
                model_name: str,
                optimizer: torch.optim.Optimizer,
                criterion: Union[Callable, torch.nn.Module],
                checkpoint_dir: str,
                checkpoint_metric: str,
                epochs: int,
                device: str,
                metrics: dict = {}
                ) -> None:
    """
    An example of ignite & pytorch training.
    :param train_loader:
    :param val_loader:
    :param dataset_name:
    :param model:
    :param model_name:
    :param optimizer:
    :param criterion:
    :param metrics:
    :param checkpoint_dir:
    :param checkpoint_metric:
    :param epochs:
    :param device:
    :return: None
    """
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    available_metrics = {
        'loss': Loss(criterion),
    }
    available_metrics.update(metrics)

    train_evaluator = create_supervised_evaluator(model, metrics=available_metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=available_metrics, device=device)

    training_history = {k: [] for k in available_metrics.keys()}
    validation_history = {k: [] for k in available_metrics.keys()}
    last_epoch = []

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        for metric_name in available_metrics.keys():
            training_history[metric_name].append(metrics[metric_name])
        last_epoch.append(0)
        print(f"Training Results - Epoch: {trainer.state.epoch}  Metrics: {metrics}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        for metric_name in available_metrics.keys():
            validation_history[metric_name].append(metrics[metric_name])
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Metrics: {metrics}")

    def get_last_val_score(eng):
        score = validation_history[checkpoint_metric][-1]
        if checkpoint_metric == 'loss':
            score = 1.0 / score
        return score

    get_last_epoch = lambda eng, last_state: eng.state.epoch
    checkpointer = ModelCheckpoint(checkpoint_dir,
                                   model_name,
                                   global_step_transform=get_last_epoch,
                                   score_function=get_last_val_score,
                                   create_dir=True,
                                   require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {dataset_name: model})

    tb_logger = tbl.TensorboardLogger(log_dir=os.path.join(checkpoint_dir, 'tboard'))
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        log_handler=tbl.GradsHistHandler(model)
    )
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss}
    )
    tb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=list(available_metrics.keys()),
        global_step_transform=global_step_from_engine(trainer),
    )

    ProgressBar().attach(trainer)
    ProgressBar().attach(train_evaluator)
    ProgressBar().attach(val_evaluator)

    trainer.run(train_loader, max_epochs=epochs)


class YOLOLoss(nn.Module):

    def __init__(self, feature_size: int, num_bboxes: int, num_classes: int, lambda_coord=5.0, lambda_noobj=0.5):
        """
        Source: https://github.com/motokimura/yolo_v1_pytorch

        :param feature_size: (int) size of input feature map.
        :param num_bboxes: (int) number of bboxes per each cell.
        :param num_classes: (int) number of the object classes.
        :param lambda_coord: (float) weight for bbox location/size losses.
        :param lambda_noobj: (float) weight for no-objectness loss.
        """
        super(YOLOLoss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt  # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])  # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])  # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter  # [N, M, 2]
        iou = inter / union  # [N, M, 2]

        return iou

    @staticmethod
    def _bbox2xyxy(bbox: torch.Tensor, s: int) -> torch.Tensor:
        bbox_xyxy = Variable(torch.FloatTensor(bbox.size()))
        bbox_xyxy[:, :2] = bbox[:, :2] / float(s) - 0.5 * bbox[:, 2:4]
        bbox_xyxy[:, 2:4] = bbox[:, :2] / float(s) + 0.5 * bbox[:, 2:4]
        return bbox_xyxy

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """

        S, B, C = self.S, self.B, self.C
        N = 5 * B + C  # 5=len([x, y, w, h, conf]

        batch_size = pred_tensor.size(0)
        coord_mask = target_tensor[:, :, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
        noobj_mask = target_tensor[:, :, :, 4] == 0  # mask for the cells which do not contain objects. [n_batch, S, S]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)  # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)  # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_pred = pred_tensor[coord_mask].view(-1, N)  # pred tensor on the cells which contain objects. [n_coord, N]
        # n_coord: number of the cells which contain objects.
        bbox_pred = coord_pred[:, :5 * B].contiguous().view(-1, 5)  # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_pred = coord_pred[:, 5 * B:]  # [n_coord, C]

        coord_target = target_tensor[coord_mask].view(-1,
                                                      N)  # target tensor on the cells which contain objects. [n_coord, N]
        # n_coord: number of the cells which contain objects.
        bbox_target = coord_target[:, :5 * B].contiguous().view(-1, 5)  # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_target = coord_target[:, 5 * B:]  # [n_coord, C]

        # Compute loss for the cells with no object bbox.
        noobj_pred = pred_tensor[noobj_mask].view(-1,
                                                  N)  # pred tensor on the cells which do not contain objects. [n_noobj, N]
        # n_noobj: number of the cells which do not contain objects.
        noobj_target = target_tensor[noobj_mask].view(-1,
                                                      N)  # target tensor on the cells which do not contain objects. [n_noobj, N]
        # n_noobj: number of the cells which do not contain objects.
        noobj_conf_mask = torch.cuda.ByteTensor(noobj_pred.size()).fill_(0)  # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask[:, 4 + b * 5] = 1  # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
            noobj_conf_mask = noobj_conf_mask.type(torch.bool)
        noobj_pred_conf = noobj_pred[noobj_conf_mask]  # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]  # [n_noobj, 2=len([conf1, conf2])]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(0)  # [n_coord x B, 5]
        coord_not_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(1)  # [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()  # [n_coord x B, 5], only the last 1=(conf,) is used

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i + B]  # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = self._bbox2xyxy(pred, S)
            # pred_xyxy = Variable(torch.FloatTensor(pred.size()))  # [B, 5=len([x1, y1, x2, y2, conf])]
            # # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            # pred_xyxy[:, :2] = pred[:, :2] / float(S) - 0.5 * pred[:, 2:4]
            # pred_xyxy[:, 2:4] = pred[:, :2] / float(S) + 0.5 * pred[:, 2:4]

            # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = bbox_target[i].view(-1, 5)  # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            # target_xyxy = Variable(torch.FloatTensor(target.size()))  # [1, 5=len([x1, y1, x2, y2, conf])]
            # # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            # target_xyxy[:, :2] = target[:, :2] / float(S) - 0.5 * target[:, 2:4]
            # target_xyxy[:, 2:4] = target[:, :2] / float(S) + 0.5 * target[:, 2:4]
            target_xyxy = self._bbox2xyxy(target, S)

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])  # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i + max_index] = 1
            coord_not_response_mask[i + max_index] = 0
            coord_response_mask = coord_response_mask.type(torch.bool)
            coord_not_response_mask = coord_not_response_mask.type(torch.bool)

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        bbox_target_iou = Variable(bbox_target_iou).cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)  # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1,
                                                                     5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1,
                                                               5)  # [n_response, 5], only the last 1=(conf,) is used
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]),
                             reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss
