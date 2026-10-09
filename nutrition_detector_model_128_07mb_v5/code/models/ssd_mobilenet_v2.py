"""MobileNetV2 SSD model builder."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config as cfg


def _sep_conv_bn_relu(x, filters, kernel=3, stride=1, name=""):
    x = layers.SeparableConv2D(
        filters,
        kernel,
        strides=stride,
        padding="same",
        use_bias=False,
        name=name + "_sepconv",
    )(x)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    return layers.ReLU(6.0, name=name + "_relu")(x)


def _conv_bn_relu(x, filters, kernel=1, stride=1, name=""):
    x = layers.Conv2D(
        filters,
        kernel,
        strides=stride,
        padding="same",
        use_bias=False,
        name=name + "_conv",
    )(x)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    return layers.ReLU(6.0, name=name + "_relu")(x)


def _build_fpn_lite(c4, c5, fpn_channels):
    if cfg.NUM_FEATURE_MAPS < 2:
        raise ValueError("FEATURE_MAP_SIZES must contain at least C4 and C5")

    p4_lat = _conv_bn_relu(c4, fpn_channels, name="fpn_lat_c4")
    p5_lat = _conv_bn_relu(c5, fpn_channels, name="fpn_lat_c5")
    p5_up = layers.UpSampling2D(
        size=2,
        interpolation="nearest",
        name="fpn_upsample_p5",
    )(p5_lat)
    p4 = layers.Add(name="fpn_add_p4")([p4_lat, p5_up])
    p4 = _sep_conv_bn_relu(p4, fpn_channels, name="fpn_refine_p4")
    p5 = _sep_conv_bn_relu(p5_lat, fpn_channels, name="fpn_refine_p5")

    feature_maps = [p4, p5]
    previous = p5
    for extra_index in range(cfg.NUM_FEATURE_MAPS - 2):
        level = 6 + extra_index
        previous = _sep_conv_bn_relu(
            previous,
            fpn_channels,
            stride=2,
            name=f"fpn_extra_p{level}",
        )
        feature_maps.append(previous)
    return feature_maps


def build_ssd_model(
    alpha: float = 1.0,
    fpn_channels: int = 256,
    num_classes: int = cfg.NUM_CLASSES,
    image_size: int = cfg.IMAGE_SIZE,
    pretrained: bool = True,
    return_features: bool = True,
    name: str = "ssd_mobilenetv2",
):
    weights = "imagenet" if pretrained else None
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            alpha=alpha,
            include_top=False,
            weights=weights,
        )
    except Exception as exc:
        if not pretrained:
            raise
        print(
            "  WARNING: ImageNet weights could not be loaded; "
            f"using random backbone initialization ({exc})"
        )
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            alpha=alpha,
            include_top=False,
            weights=None,
        )

    c4 = backbone.get_layer(cfg.BACKBONE_C4_LAYER).output
    c5 = backbone.get_layer(cfg.BACKBONE_C5_LAYER).output
    feature_maps = _build_fpn_lite(c4, c5, fpn_channels)

    anchors_per_location = cfg.NUM_ANCHORS_PER_LOC
    all_cls, all_box = [], []
    foreground_prior = 0.01
    class_prior = np.full(
        num_classes,
        foreground_prior / max(num_classes - 1, 1),
        dtype=np.float32,
    )
    class_prior[0] = 1.0 - foreground_prior
    cls_bias = np.tile(np.log(class_prior), anchors_per_location)

    for level, feature in enumerate(feature_maps):
        cls = _sep_conv_bn_relu(feature, fpn_channels, name=f"cls_head_l{level}")
        cls = layers.Conv2D(
            anchors_per_location * num_classes,
            1,
            padding="same",
            bias_initializer=tf.keras.initializers.Constant(cls_bias),
            name=f"cls_pred_l{level}",
        )(cls)
        box = _sep_conv_bn_relu(feature, fpn_channels, name=f"box_head_l{level}")
        box = layers.Conv2D(
            anchors_per_location * 4,
            1,
            padding="same",
            name=f"box_pred_l{level}",
        )(box)
        cls = layers.Reshape((-1, num_classes), name=f"cls_reshape_l{level}")(cls)
        box = layers.Reshape((-1, 4), name=f"box_reshape_l{level}")(box)
        all_cls.append(cls)
        all_box.append(box)

    cls_preds = layers.Concatenate(axis=1, name="cls_preds")(all_cls)
    box_preds = layers.Concatenate(axis=1, name="box_preds")(all_box)
    outputs = {"cls_preds": cls_preds, "box_preds": box_preds}
    if return_features:
        for index, feature in enumerate(feature_maps):
            outputs[f"feat_{index}"] = feature

    return tf.keras.Model(inputs=backbone.input, outputs=outputs, name=name)


def build_teacher(return_features=True, pretrained=True):
    return build_ssd_model(
        alpha=cfg.TEACHER_ALPHA,
        fpn_channels=cfg.TEACHER_FPN_CHANNELS,
        pretrained=pretrained,
        return_features=return_features,
        name="teacher_ssd",
    )
