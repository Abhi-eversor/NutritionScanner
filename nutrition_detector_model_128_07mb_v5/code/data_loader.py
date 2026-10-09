"""TensorFlow graph input pipeline for Roboflow TFRecords."""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

import config as cfg
from anchor_utils import ANCHORS_TF, NUM_ANCHORS, match_anchors_to_gt


_FEATURE_DESCRIPTION = {
    "image/encoded": tf.io.FixedLenFeature([], tf.string),
    "image/height": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "image/width": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
}


def _parse_example(serialized):
    feats = tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION)
    image = tf.cast(
        tf.io.decode_jpeg(feats["image/encoded"], channels=3),
        tf.float32,
    )

    xmin = tf.sparse.to_dense(feats["image/object/bbox/xmin"])
    ymin = tf.sparse.to_dense(feats["image/object/bbox/ymin"])
    xmax = tf.sparse.to_dense(feats["image/object/bbox/xmax"])
    ymax = tf.sparse.to_dense(feats["image/object/bbox/ymax"])

    num_boxes = tf.reduce_min(
        tf.stack([
            tf.size(xmin),
            tf.size(ymin),
            tf.size(xmax),
            tf.size(ymax),
        ])
    )
    boxes = tf.stack(
        [
            xmin[:num_boxes],
            ymin[:num_boxes],
            xmax[:num_boxes],
            ymax[:num_boxes],
        ],
        axis=1,
    )
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)
    labels = tf.ones([num_boxes], dtype=tf.int32)

    finite = tf.reduce_all(tf.math.is_finite(boxes), axis=1)
    valid = (
        finite
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes = tf.boolean_mask(boxes, valid)
    labels = tf.boolean_mask(labels, valid)
    return image, boxes, labels


def _crop_annotations(image, boxes, labels):
    image_shape = tf.shape(image)
    image_h = tf.cast(image_shape[0], tf.float32)
    image_w = tf.cast(image_shape[1], tf.float32)

    # sample_distorted_bounding_box expects [ymin, xmin, ymax, xmax].
    boxes_yxyx = tf.stack(
        [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]],
        axis=1,
    )
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        image_shape,
        bounding_boxes=boxes_yxyx[tf.newaxis, ...],
        min_object_covered=0.5,
        aspect_ratio_range=(0.5, 2.0),
        area_range=(0.3, 1.0),
        max_attempts=50,
        use_image_if_no_bounding_boxes=True,
    )
    cropped = tf.slice(image, begin, size)

    y0 = tf.cast(begin[0], tf.float32) / image_h
    x0 = tf.cast(begin[1], tf.float32) / image_w
    crop_h = tf.cast(size[0], tf.float32) / image_h
    crop_w = tf.cast(size[1], tf.float32) / image_w

    new_boxes = tf.stack(
        [
            (boxes[:, 0] - x0) / crop_w,
            (boxes[:, 1] - y0) / crop_h,
            (boxes[:, 2] - x0) / crop_w,
            (boxes[:, 3] - y0) / crop_h,
        ],
        axis=1,
    )
    new_boxes = tf.clip_by_value(new_boxes, 0.0, 1.0)
    valid = (
        ((new_boxes[:, 2] - new_boxes[:, 0]) > 0.01)
        & ((new_boxes[:, 3] - new_boxes[:, 1]) > 0.01)
    )
    new_boxes = tf.boolean_mask(new_boxes, valid)
    new_labels = tf.boolean_mask(labels, valid)

    return tf.cond(
        tf.size(new_labels) > 0,
        lambda: (cropped, new_boxes, new_labels),
        lambda: (image, boxes, labels),
    )


def _random_crop(image, boxes, labels):
    should_crop = (
        (tf.random.uniform([]) < cfg.RANDOM_CROP_PROB)
        & (tf.shape(boxes)[0] > 0)
    )
    return tf.cond(
        should_crop,
        lambda: _crop_annotations(image, boxes, labels),
        lambda: (image, boxes, labels),
    )


def _random_horizontal_flip(image, boxes):
    def flip():
        flipped_boxes = tf.stack(
            [
                1.0 - boxes[:, 2],
                boxes[:, 1],
                1.0 - boxes[:, 0],
                boxes[:, 3],
            ],
            axis=1,
        )
        return tf.image.flip_left_right(image), flipped_boxes

    return tf.cond(
        tf.random.uniform([]) < cfg.RANDOM_HORIZONTAL_FLIP_PROB,
        flip,
        lambda: (image, boxes),
    )


def _random_color_jitter(image):
    image = tf.image.random_brightness(image, max_delta=32.0)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.05)
    return tf.clip_by_value(image, 0.0, 255.0)


def _finish_preprocessing(image, boxes, labels):
    image = tf.image.resize(
        image,
        [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE],
        antialias=True,
    )
    image = tf.clip_by_value(image / 127.5 - 1.0, -1.0, 1.0)

    cls_targets, box_targets = match_anchors_to_gt(
        boxes, labels, ANCHORS_TF
    )
    image = tf.ensure_shape(
        image, [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3]
    )
    cls_targets = tf.ensure_shape(cls_targets, [NUM_ANCHORS])
    box_targets = tf.ensure_shape(box_targets, [NUM_ANCHORS, 4])
    return image, cls_targets, box_targets


def _preprocess_train(image, boxes, labels):
    image, boxes, labels = _random_crop(image, boxes, labels)
    image, boxes = _random_horizontal_flip(image, boxes)
    image = tf.cond(
        tf.random.uniform([]) < cfg.RANDOM_COLOR_JITTER_PROB,
        lambda: _random_color_jitter(image),
        lambda: image,
    )
    return _finish_preprocessing(image, boxes, labels)


def _preprocess_val(image, boxes, labels):
    return _finish_preprocessing(image, boxes, labels)


def _preprocess_detection_eval(image, boxes, labels):
    image = tf.image.resize(
        image,
        [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE],
        antialias=True,
    )
    image = tf.clip_by_value(image / 127.5 - 1.0, -1.0, 1.0)
    image = tf.ensure_shape(
        image, [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3]
    )
    boxes = tf.ensure_shape(boxes, [None, 4])
    labels = tf.ensure_shape(labels, [None])
    return image, boxes, labels, tf.shape(labels)[0]


def _parse_and_preprocess_train(serialized):
    return _preprocess_train(*_parse_example(serialized))


def _parse_and_preprocess_val(serialized):
    return _preprocess_val(*_parse_example(serialized))


def _parse_and_preprocess_detection_eval(serialized):
    return _preprocess_detection_eval(*_parse_example(serialized))


def _set_if_available(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def _dataset_options(is_training):
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    options.deterministic = not is_training

    optimization = options.experimental_optimization
    _set_if_available(optimization, "map_parallelization", True)
    _set_if_available(optimization, "map_and_batch_fusion", True)
    _set_if_available(optimization, "noop_elimination", True)
    _set_if_available(optimization, "parallel_batch", True)

    return options


def create_dataset(tfrecord_path, is_training=True, batch_size=None):
    if batch_size is None:
        batch_size = cfg.GLOBAL_BATCH_SIZE
    if not tf.io.gfile.exists(tfrecord_path):
        raise FileNotFoundError(
            f"TFRecord not found: {tfrecord_path}. "
            "Set SSD_DATA_DIR to the folder containing train/valid/test."
        )

    ds = tf.data.TFRecordDataset(
        tfrecord_path,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    if is_training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
        ds = ds.repeat()

    ds = ds.map(
        (
            _parse_and_preprocess_train
            if is_training
            else _parse_and_preprocess_val
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not is_training,
    )
    ds = ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.with_options(_dataset_options(is_training))


def create_detection_eval_dataset(tfrecord_path, batch_size=1):
    if not tf.io.gfile.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    ds = tf.data.TFRecordDataset(
        tfrecord_path,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        _parse_and_preprocess_detection_eval,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3],
            [None, 4],
            [None],
            [],
        ),
        padding_values=(
            tf.constant(0.0, tf.float32),
            tf.constant(0.0, tf.float32),
            tf.constant(0, tf.int32),
            tf.constant(0, tf.int32),
        ),
        drop_remainder=False,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds.with_options(_dataset_options(is_training=False))


def count_records(tfrecord_path):
    if not tf.io.gfile.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")
    return sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
