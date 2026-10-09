"""Structured pruning for the MobileNetV2 SSD student."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config as cfg


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_block_table(alpha):
    raw_blocks = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    first_filters = _make_divisible(32 * alpha)
    blocks = []
    c_in = first_filters
    block_id = 0

    for t, c_raw, n, s in raw_blocks:
        c_out = _make_divisible(c_raw * alpha)
        for repeat_idx in range(n):
            stride = s if repeat_idx == 0 else 1
            has_residual = stride == 1 and c_in == c_out
            expansion_channels = _make_divisible(t * c_in) if t > 1 else c_in
            blocks.append(
                {
                    "block_id": block_id,
                    "t": t,
                    "c_in": c_in,
                    "c_out": c_out,
                    "stride": stride,
                    "has_residual": has_residual,
                    "expansion_channels": expansion_channels,
                }
            )
            c_in = c_out
            block_id += 1

    return blocks


def analyse_expansion_importance(model, alpha=None):
    if alpha is None:
        alpha = cfg.STUDENT_ALPHA
    importance = {}

    for block in get_block_table(alpha):
        block_id = block["block_id"]
        if block["t"] == 1:
            continue

        # Keras names the first MobileNetV2 block differently.
        layer_name = (
            "expanded_conv_expand"
            if block_id == 0
            else f"block_{block_id}_expand"
        )
        try:
            layer = model.get_layer(layer_name)
        except ValueError:
            continue

        weights = layer.get_weights()[0]
        importance[block_id] = np.sum(np.abs(weights), axis=(0, 1, 2))

    return importance


def compute_keep_indices(importance, prune_ratio=None, min_channels=None):
    if prune_ratio is None:
        prune_ratio = cfg.EXPANSION_PRUNE_RATIO
    if min_channels is None:
        min_channels = cfg.MIN_EXPANSION_CHANNELS

    keep = {}
    new_expansion = {}
    for block_id, scores in importance.items():
        total = len(scores)
        count = max(
            min_channels,
            _make_divisible(int(total * (1.0 - prune_ratio)), 8),
        )
        count = min(count, total)
        indices = np.sort(np.argsort(scores)[::-1][:count])
        keep[block_id] = indices
        new_expansion[block_id] = count

    return keep, new_expansion


def build_custom_mobilenet_v2(block_table, first_filters, input_shape=None):
    if input_shape is None:
        input_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)

    inputs = layers.Input(shape=input_shape, name="input")
    x = layers.Conv2D(
        first_filters,
        3,
        strides=2,
        padding="same",
        use_bias=False,
        name="Conv1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_Conv1")(x)
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    for block in block_table:
        block_id = block["block_id"]
        expansion_channels = block["expansion_channels"]
        c_out = block["c_out"]
        stride = block["stride"]
        has_residual = block["has_residual"]
        expansion_factor = block["t"]
        prefix = "expanded_conv_" if block_id == 0 else f"block_{block_id}_"

        shortcut = x
        if expansion_factor > 1:
            x = layers.Conv2D(
                expansion_channels,
                1,
                use_bias=False,
                name=prefix + "expand",
            )(x)
            x = layers.BatchNormalization(name=prefix + "expand_BN")(x)
            x = layers.ReLU(6.0, name=prefix + "expand_relu")(x)

        x = layers.DepthwiseConv2D(
            3,
            strides=stride,
            padding="same",
            use_bias=False,
            name=prefix + "depthwise",
        )(x)
        x = layers.BatchNormalization(name=prefix + "depthwise_BN")(x)
        x = layers.ReLU(6.0, name=prefix + "depthwise_relu")(x)
        x = layers.Conv2D(
            c_out,
            1,
            use_bias=False,
            name=prefix + "project",
        )(x)
        x = layers.BatchNormalization(name=prefix + "project_BN")(x)

        if has_residual:
            x = layers.Add(name=prefix + "add")([shortcut, x])

    x = layers.Conv2D(1280, 1, use_bias=False, name="Conv_1")(x)
    x = layers.BatchNormalization(name="bn_Conv_1")(x)
    x = layers.ReLU(6.0, name="out_relu")(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="pruned_mobilenetv2")


def _transfer_conv_weights(src_model, dst_model, layer_name, axis, keep_idx=None):
    try:
        src_layer = src_model.get_layer(layer_name)
        dst_layer = dst_model.get_layer(layer_name)
    except ValueError:
        return

    weights = src_layer.get_weights()[0]
    if keep_idx is not None:
        weights = np.take(weights, keep_idx, axis=axis)
    dst_layer.set_weights([weights])


def _transfer_bn_weights(src_model, dst_model, layer_name, keep_idx=None):
    try:
        src_layer = src_model.get_layer(layer_name)
        dst_layer = dst_model.get_layer(layer_name)
    except ValueError:
        return

    params = src_layer.get_weights()
    if keep_idx is not None:
        params = [param[keep_idx] for param in params]
    dst_layer.set_weights(params)


def transfer_backbone_weights(src_model, dst_model, block_table, keep_indices):
    _transfer_conv_weights(src_model, dst_model, "Conv1", axis=None)
    _transfer_bn_weights(src_model, dst_model, "bn_Conv1")

    for block in block_table:
        block_id = block["block_id"]
        prefix = "expanded_conv_" if block_id == 0 else f"block_{block_id}_"
        keep_idx = keep_indices.get(block_id)

        if block["t"] > 1:
            _transfer_conv_weights(
                src_model,
                dst_model,
                prefix + "expand",
                axis=3,
                keep_idx=keep_idx,
            )
            _transfer_bn_weights(
                src_model,
                dst_model,
                prefix + "expand_BN",
                keep_idx=keep_idx,
            )

        # Depthwise kernels store channels on axis 2.
        _transfer_conv_weights(
            src_model,
            dst_model,
            prefix + "depthwise",
            axis=2,
            keep_idx=keep_idx,
        )
        _transfer_bn_weights(
            src_model,
            dst_model,
            prefix + "depthwise_BN",
            keep_idx=keep_idx,
        )
        _transfer_conv_weights(
            src_model,
            dst_model,
            prefix + "project",
            axis=2,
            keep_idx=keep_idx,
        )
        _transfer_bn_weights(src_model, dst_model, prefix + "project_BN")

    _transfer_conv_weights(src_model, dst_model, "Conv_1", axis=None)
    _transfer_bn_weights(src_model, dst_model, "bn_Conv_1")


def build_pruned_ssd(
    pruned_backbone,
    fpn_channels=None,
    num_classes=None,
    return_features=False,
):
    from models.ssd_mobilenet_v2 import _build_fpn_lite, _sep_conv_bn_relu

    if fpn_channels is None:
        fpn_channels = cfg.STUDENT_FPN_CHANNELS
    if num_classes is None:
        num_classes = cfg.NUM_CLASSES

    c4 = pruned_backbone.get_layer(cfg.BACKBONE_C4_LAYER).output
    c5 = pruned_backbone.get_layer(cfg.BACKBONE_C5_LAYER).output
    feature_maps = _build_fpn_lite(c4, c5, fpn_channels)
    anchors_per_location = cfg.NUM_ANCHORS_PER_LOC
    all_cls, all_box = [], []

    for level, feature in enumerate(feature_maps):
        cls = _sep_conv_bn_relu(feature, fpn_channels, name=f"cls_head_l{level}")
        cls = layers.Conv2D(
            anchors_per_location * num_classes,
            1,
            padding="same",
            name=f"cls_pred_l{level}",
        )(cls)
        box = _sep_conv_bn_relu(feature, fpn_channels, name=f"box_head_l{level}")
        box = layers.Conv2D(
            anchors_per_location * 4,
            1,
            padding="same",
            name=f"box_pred_l{level}",
        )(box)
        cls = layers.Reshape(
            (-1, num_classes),
            name=f"cls_reshape_l{level}",
        )(cls)
        box = layers.Reshape((-1, 4), name=f"box_reshape_l{level}")(box)
        all_cls.append(cls)
        all_box.append(box)

    cls_preds = layers.Concatenate(axis=1, name="cls_preds")(all_cls)
    box_preds = layers.Concatenate(axis=1, name="box_preds")(all_box)
    outputs = {"cls_preds": cls_preds, "box_preds": box_preds}
    if return_features:
        for index, feature in enumerate(feature_maps):
            outputs[f"feat_{index}"] = feature

    return tf.keras.Model(
        inputs=pruned_backbone.input,
        outputs=outputs,
        name="pruned_student_ssd",
    )


def transfer_fpn_and_head_weights(src_model, dst_model):
    prefixes = ("fpn_", "cls_", "box_")
    for dst_layer in dst_model.layers:
        if not any(dst_layer.name.startswith(prefix) for prefix in prefixes):
            continue
        try:
            src_layer = src_model.get_layer(dst_layer.name)
            if src_layer.get_weights():
                dst_layer.set_weights(src_layer.get_weights())
        except (ValueError, Exception):
            pass


def prune_student(trained_student, prune_ratio=None, alpha=None, return_features=False):
    if prune_ratio is None:
        prune_ratio = cfg.EXPANSION_PRUNE_RATIO
    if alpha is None:
        alpha = cfg.STUDENT_ALPHA

    before_params = trained_student.count_params()
    table = get_block_table(alpha)
    first_filters = _make_divisible(32 * alpha)

    importance = analyse_expansion_importance(trained_student, alpha)
    print(f"\n  Analysed {len(importance)} blocks with expansion layers")
    for block_id, values in importance.items():
        print(
            f"    block {block_id:>2d}:  {len(values):>3d} expansion channels  "
            f"(L1 range {values.min():.4f} to {values.max():.4f})"
        )

    keep_indices, new_expansion = compute_keep_indices(importance, prune_ratio)
    print(f"\n  Pruning plan (ratio={prune_ratio}):")
    for block_id in keep_indices:
        original = len(importance[block_id])
        kept = new_expansion[block_id]
        print(
            f"    block {block_id:>2d}:  {original} -> {kept}  "
            f"(removed {original - kept})"
        )

    for block in table:
        block_id = block["block_id"]
        if block_id in new_expansion:
            block["expansion_channels"] = new_expansion[block_id]

    pruned_backbone = build_custom_mobilenet_v2(table, first_filters)
    transfer_backbone_weights(
        trained_student,
        pruned_backbone,
        table,
        keep_indices,
    )
    pruned_ssd = build_pruned_ssd(
        pruned_backbone,
        return_features=return_features,
    )
    transfer_fpn_and_head_weights(trained_student, pruned_ssd)

    after_params = pruned_ssd.count_params()
    stats = {
        "before_params": before_params,
        "after_params": after_params,
        "reduction": 1.0 - after_params / before_params,
    }
    print("\n  Pruning complete")
    print(
        f"    Parameters:  {before_params:,} -> {after_params:,}  "
        f"({stats['reduction']:.1%} reduction)"
    )
    return pruned_ssd, stats
