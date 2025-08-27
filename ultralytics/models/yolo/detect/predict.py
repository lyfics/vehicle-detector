# # Ultralytics YOLO 🚀, AGPL-3.0 license
#
# import torch
#
# from ultralytics.engine.predictor import BasePredictor
# from ultralytics.engine.results import Results
# from ultralytics.utils import DEFAULT_CFG, ROOT, ops
#
#
#
#
# class DetectionPredictor(BasePredictor):
#
#     def postprocess(self, preds, img, orig_imgs):
#         """Postprocesses predictions and returns a list of Results objects."""
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det,
#                                         classes=self.args.classes)
#
#         results = []
#         for i, pred in enumerate(preds):
#             orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
#             if not isinstance(orig_imgs, torch.Tensor):
#                 pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
#             path = self.batch[0]
#             img_path = path[i] if isinstance(path, list) else path
#             results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
#         return results
#
#
# def predict(cfg=DEFAULT_CFG, use_python=False):
#     """Runs YOLO model inference on input image(s)."""
#     model = cfg.model or 'best.pt'
#     source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
#         else 'https://ultralytics.com/images/bus.jpg'
#
#     args = dict(model=model, source=source)
#     if use_python:
#         from ultralytics import YOLO
#         YOLO(model)(**args)
#     else:
#         predictor = DetectionPredictor(overrides=args)
#         predictor.predict_cli()
#
#
# if __name__ == '__main__':
#     predict()
# Ultralytics YOLO , AGPL-3.0 license

import torch
from typing import Dict, Tuple

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ROOT, ops


class VehicleCounter:

    def __init__(self, line_type: str = 'horizontal'):
        """
        Vehicle counting system
        :param line_type: 'horizontal' or 'vertical' counting line
        """
        self.line_type = line_type.lower()
        self.count_left_to_right = 0
        self.count_right_to_left = 0
        self.prev_positions: Dict[int, Tuple[float, float]] = {}  # {track_id: (x_center, y_center)}

    def update_count(self, box: torch.Tensor, track_id: int, frame_width: int, frame_height: int) -> None:
        """
        Update vehicle counts based on crossing events
        :param box: Detection box [x1, y1, x2, y2]
        :param track_id: Vehicle track ID
        :param frame_width: Frame width
        :param frame_height: Frame height
        """
        x1, y1, x2, y2 = box
        current_pos = ((x1 + x2) / 2, (y1 + y2) / 2)  # Current center point

        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = current_pos
            return

        prev_x, prev_y = self.prev_positions[track_id]
        curr_x, curr_y = current_pos

        if self.line_type == 'horizontal':
            # Horizontal counting line (vertical line at center)
            line_pos = frame_width // 2
            if (prev_x < line_pos and curr_x > line_pos):
                self.count_left_to_right += 1
            elif (prev_x > line_pos and curr_x < line_pos):
                self.count_right_to_left += 1
        else:
            # Vertical counting line (horizontal line at center)
            line_pos = frame_height // 2
            if (prev_y < line_pos and curr_y > line_pos):
                self.count_left_to_right += 1  # Top to bottom
            elif (prev_y > line_pos and curr_y < line_pos):
                self.count_right_to_left += 1  # Bottom to top

        self.prev_positions[track_id] = current_pos


class DetectionPredictor(BasePredictor):

    # def __init__(self, overrides=None):
    #     super().__init__(overrides)
    #     self.vehicle_counter = VehicleCounter(line_type='horizontal')  # Initialize counter

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        return super().write_results(idx, results, batch)


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'best.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()