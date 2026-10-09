"""SSD anchor and post-processing utilities."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

import config as cfg


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = (boxes[..., i] for i in range(4))
    return tf.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = (boxes[..., i] for i in range(4))
    return tf.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)


def generate_anchors():
    fmap_sizes = cfg.FEATURE_MAP_SIZES
    num_levels = len(fmap_sizes)
    scales = np.linspace(cfg.ANCHOR_SCALE_MIN, cfg.ANCHOR_SCALE_MAX, num_levels)
    scales_ext = np.append(scales, 1.05)
    extra_scales = np.sqrt(scales * scales_ext[1:])

    anchors = []
    for level, fmap in enumerate(fmap_sizes):
        s = scales[level]
        s_e = extra_scales[level]
        for i in range(fmap):
            cy = (i + 0.5) / fmap
            for j in range(fmap):
                cx = (j + 0.5) / fmap
                anchors.append([cx, cy, s, s])
                anchors.append([cx, cy, s_e, s_e])
                for ar in cfg.ASPECT_RATIOS:
                    if ar == 1.0:
                        continue
                    w = s * np.sqrt(ar)
                    h = s / np.sqrt(ar)
                    anchors.append([cx, cy, w, h])

    return np.asarray(anchors, dtype=np.float32)


ANCHORS_NP = generate_anchors()
ANCHORS_TF = tf.constant(ANCHORS_NP)
NUM_ANCHORS = ANCHORS_NP.shape[0]


def compute_iou(boxes_a, boxes_b):
    a = tf.expand_dims(boxes_a, 1)
    b = tf.expand_dims(boxes_b, 0)
    inter_min = tf.maximum(a[..., :2], b[..., :2])
    inter_max = tf.minimum(a[..., 2:], b[..., 2:])
    inter_wh = tf.maximum(inter_max - inter_min, 0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / tf.maximum(area_a + area_b - inter, 1e-7)


def encode_boxes(gt_cxcywh, anchor_cxcywh, variances=None):
    if variances is None:
        variances = cfg.BOX_VARIANCE

    dcx = (gt_cxcywh[..., 0] - anchor_cxcywh[..., 0]) / anchor_cxcywh[..., 2] / variances[0]
    dcy = (gt_cxcywh[..., 1] - anchor_cxcywh[..., 1]) / anchor_cxcywh[..., 3] / variances[1]
    dw = tf.math.log(gt_cxcywh[..., 2] / anchor_cxcywh[..., 2] + 1e-7) / variances[2]
    dh = tf.math.log(gt_cxcywh[..., 3] / anchor_cxcywh[..., 3] + 1e-7) / variances[3]
    return tf.stack([dcx, dcy, dw, dh], axis=-1)


def decode_boxes(pred_offsets, anchor_cxcywh=None, variances=None):
    if anchor_cxcywh is None:
        anchor_cxcywh = ANCHORS_TF
    if variances is None:
        variances = cfg.BOX_VARIANCE

    cx = pred_offsets[..., 0] * variances[0] * anchor_cxcywh[..., 2] + anchor_cxcywh[..., 0]
    cy = pred_offsets[..., 1] * variances[1] * anchor_cxcywh[..., 3] + anchor_cxcywh[..., 1]
    w = tf.exp(pred_offsets[..., 2] * variances[2]) * anchor_cxcywh[..., 2]
    h = tf.exp(pred_offsets[..., 3] * variances[3]) * anchor_cxcywh[..., 3]
    return tf.stack([cx, cy, w, h], axis=-1)


def _match_with_gt(gt_boxes_xyxy, gt_labels, anchors_cxcywh):
    num_anchors = tf.shape(anchors_cxcywh)[0]
    anchors_xyxy = cxcywh_to_xyxy(anchors_cxcywh)
    iou_matrix = compute_iou(gt_boxes_xyxy, anchors_xyxy)
    best_gt_iou = tf.reduce_max(iou_matrix, axis=0)
    best_gt_idx = tf.cast(tf.argmax(iou_matrix, axis=0), tf.int32)
    best_anchor_idx = tf.cast(tf.argmax(iou_matrix, axis=1), tf.int32)

    cls_targets = tf.zeros([num_anchors], dtype=tf.int32)
    ignore_mask = (best_gt_iou >= cfg.IOU_NEG_THRESHOLD) & (best_gt_iou < cfg.IOU_POS_THRESHOLD)
    cls_targets = tf.where(ignore_mask, -tf.ones_like(cls_targets), cls_targets)
    pos_mask = best_gt_iou >= cfg.IOU_POS_THRESHOLD
    cls_targets = tf.where(pos_mask, tf.gather(gt_labels, best_gt_idx), cls_targets)

    # Ensure every ground-truth box has a positive anchor.
    cls_targets = tf.tensor_scatter_nd_update(
        cls_targets,
        tf.expand_dims(best_anchor_idx, 1),
        gt_labels,
    )
    gt_indices_for_forced = tf.range(tf.shape(gt_labels)[0], dtype=tf.int32)
    best_gt_idx = tf.tensor_scatter_nd_update(
        best_gt_idx,
        tf.expand_dims(best_anchor_idx, 1),
        gt_indices_for_forced,
    )

    gt_cxcywh = xyxy_to_cxcywh(gt_boxes_xyxy)
    matched_gt = tf.gather(gt_cxcywh, best_gt_idx)
    box_targets = encode_boxes(matched_gt, anchors_cxcywh)
    pos_float = tf.cast(cls_targets > 0, tf.float32)
    box_targets = box_targets * tf.expand_dims(pos_float, -1)
    return cls_targets, box_targets


def _empty_match(anchors_cxcywh):
    return (
        tf.zeros([NUM_ANCHORS], dtype=tf.int32),
        tf.zeros([NUM_ANCHORS, 4], dtype=tf.float32),
    )


def match_anchors_to_gt(gt_boxes_xyxy, gt_labels, anchors_cxcywh=None):
    if anchors_cxcywh is None:
        anchors_cxcywh = ANCHORS_TF

    gt_labels = tf.cast(gt_labels, tf.int32)
    gt_boxes_xyxy = tf.reshape(gt_boxes_xyxy, [-1, 4])
    gt_labels = tf.reshape(gt_labels, [-1])

    # Keep box and label arrays aligned after malformed records or augmentation.
    num_gt = tf.minimum(
        tf.shape(gt_boxes_xyxy, out_type=tf.int32)[0],
        tf.shape(gt_labels, out_type=tf.int32)[0],
    )
    gt_boxes_xyxy = gt_boxes_xyxy[:num_gt]
    gt_labels = gt_labels[:num_gt]
    has_gt = tf.greater(tf.size(gt_boxes_xyxy), 0)

    return tf.cond(
        has_gt,
        lambda: _match_with_gt(gt_boxes_xyxy, gt_labels, anchors_cxcywh),
        lambda: _empty_match(anchors_cxcywh),
    )


def apply_nms(cls_preds, box_preds, score_thresh=None, iou_thresh=None, max_dets=None):
    if score_thresh is None:
        score_thresh = cfg.NMS_SCORE_THRESHOLD
    if iou_thresh is None:
        iou_thresh = cfg.NMS_IOU_THRESHOLD
    if max_dets is None:
        max_dets = cfg.MAX_DETECTIONS_PER_IMAGE

    decoded_cxcywh = decode_boxes(box_preds)
    decoded_xyxy = cxcywh_to_xyxy(decoded_cxcywh)
    decoded_xyxy = tf.clip_by_value(decoded_xyxy, 0.0, 1.0)
    scores = tf.nn.softmax(cls_preds, axis=-1)[:, 1:]
    scores = tf.squeeze(scores, axis=-1)

    selected = tf.image.non_max_suppression(
        decoded_xyxy,
        scores,
        max_output_size=max_dets,
        iou_threshold=iou_thresh,
        score_threshold=score_thresh,
    )

    sel_boxes = tf.gather(decoded_xyxy, selected)
    sel_scores = tf.gather(scores, selected)
    sel_classes = tf.ones([tf.shape(selected)[0]], dtype=tf.int32)
    num_det = tf.shape(selected)[0]
    pad_n = max_dets - num_det
    sel_boxes = tf.pad(sel_boxes, [[0, pad_n], [0, 0]])
    sel_scores = tf.pad(sel_scores, [[0, pad_n]])
    sel_classes = tf.pad(sel_classes, [[0, pad_n]])

    return sel_boxes, sel_scores, sel_classes, num_det
