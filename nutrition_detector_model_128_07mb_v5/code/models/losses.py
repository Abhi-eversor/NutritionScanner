"""SSD losses."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

import config as cfg


def focal_loss(cls_preds, cls_targets, alpha=None, gamma=None):
    if alpha is None:
        alpha = cfg.FOCAL_ALPHA
    if gamma is None:
        gamma = cfg.FOCAL_GAMMA

    num_classes = tf.shape(cls_preds)[-1]
    valid_mask = tf.cast(cls_targets >= 0, tf.float32)
    pos_mask = tf.cast(cls_targets > 0, tf.float32)
    num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
    safe_targets = tf.maximum(cls_targets, 0)
    targets_oh = tf.one_hot(safe_targets, num_classes)
    probs = tf.nn.softmax(cls_preds, axis=-1)
    p_t = tf.reduce_sum(targets_oh * probs, axis=-1)
    alpha_t = tf.where(safe_targets > 0, alpha, 1.0 - alpha)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=safe_targets,
        logits=cls_preds,
    )
    focal = alpha_t * tf.pow(1.0 - p_t, gamma) * cross_entropy
    focal = focal * valid_mask
    return tf.reduce_sum(focal) / num_pos


def smooth_l1_loss(box_preds, box_targets, cls_targets):
    pos_mask = tf.cast(cls_targets > 0, tf.float32)
    num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
    diff = tf.abs(box_preds - box_targets)
    loss = tf.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)
    loss = tf.reduce_sum(loss, axis=-1) * pos_mask
    return tf.reduce_sum(loss) / num_pos


def ssd_loss(cls_preds, box_preds, cls_targets, box_targets):
    cls_loss = focal_loss(cls_preds, cls_targets)
    loc_loss = smooth_l1_loss(box_preds, box_targets, cls_targets)
    total = cls_loss + cfg.LOC_LOSS_WEIGHT * loc_loss
    return total, cls_loss, loc_loss
