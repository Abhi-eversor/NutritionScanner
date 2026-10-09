"""Comparable AP50, precision, and recall evaluation for all model stages."""

import json
import os
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

import config as cfg
from anchor_utils import cxcywh_to_xyxy, decode_boxes
from data_loader import create_detection_eval_dataset


def _softmax(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)


def _postprocess(cls_logits, box_offsets):
    cls_logits = np.asarray(cls_logits, dtype=np.float32)
    box_offsets = np.asarray(box_offsets, dtype=np.float32)
    scores = _softmax(cls_logits)[:, 1]

    decoded = decode_boxes(tf.convert_to_tensor(box_offsets))
    boxes = cxcywh_to_xyxy(decoded)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)

    # TensorFlow NMS expects [ymin, xmin, ymax, xmax].
    nms_boxes = tf.stack(
        [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]],
        axis=1,
    )
    selected = tf.image.non_max_suppression(
        nms_boxes,
        tf.convert_to_tensor(scores),
        max_output_size=cfg.EVAL_MAX_DETECTIONS,
        iou_threshold=cfg.NMS_IOU_THRESHOLD,
        score_threshold=cfg.EVAL_AP_MIN_SCORE,
    ).numpy()
    return boxes.numpy()[selected], scores[selected]


def _extract_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs["cls_preds"], outputs["box_preds"]

    if not isinstance(outputs, (tuple, list)):
        raise TypeError(f"Unsupported model output type: {type(outputs)}")

    cls_output = box_output = None
    for output in outputs:
        if output.shape[-1] == cfg.NUM_CLASSES:
            cls_output = output
        elif output.shape[-1] == 4:
            box_output = output
    if cls_output is None or box_output is None:
        raise ValueError("Could not identify cls_preds and box_preds")
    return cls_output, box_output


class KerasPredictor:
    def __init__(self, model):
        self.model = model

    def __call__(self, images):
        outputs = self.model(images, training=False)
        cls_logits, box_offsets = _extract_outputs(outputs)
        return cls_logits.numpy(), box_offsets.numpy()


def _dequantize(array, detail):
    scale, zero_point = detail["quantization"]
    if scale == 0:
        return array.astype(np.float32)
    return (array.astype(np.float32) - zero_point) * scale


class TFLitePredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        input_shape = tuple(self.input_detail["shape"])
        expected = (1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
        if input_shape != expected:
            raise ValueError(
                f"Unexpected TFLite input shape {input_shape}; "
                f"expected {expected}"
            )

    def _quantize_input(self, images):
        images = np.asarray(images, dtype=np.float32)
        dtype = self.input_detail["dtype"]
        if np.issubdtype(dtype, np.floating):
            return images.astype(dtype)

        scale, zero_point = self.input_detail["quantization"]
        if scale == 0:
            raise ValueError("TFLite integer input has no quantization scale")
        quantized = np.round(images / scale + zero_point)
        limits = np.iinfo(dtype)
        return np.clip(quantized, limits.min, limits.max).astype(dtype)

    def __call__(self, images):
        if len(images) != 1:
            raise ValueError("TFLite evaluation requires batch_size=1")
        self.interpreter.set_tensor(
            self.input_detail["index"],
            self._quantize_input(images),
        )
        self.interpreter.invoke()

        cls_logits = box_offsets = None
        for detail in self.output_details:
            value = self.interpreter.get_tensor(detail["index"])
            value = _dequantize(value, detail)
            if value.shape[-1] == cfg.NUM_CLASSES:
                cls_logits = value
            elif value.shape[-1] == 4:
                box_offsets = value
        if cls_logits is None or box_offsets is None:
            shapes = [tuple(d["shape"]) for d in self.output_details]
            raise ValueError(
                f"Could not identify TFLite outputs from shapes {shapes}"
            )
        return cls_logits, box_offsets


def _iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty([0], dtype=np.float32)
    inter_min = np.maximum(box[:2], boxes[:, :2])
    inter_max = np.minimum(box[2:], boxes[:, 2:])
    inter_wh = np.maximum(inter_max - inter_min, 0.0)
    intersection = inter_wh[:, 0] * inter_wh[:, 1]
    area_box = max(
        (box[2] - box[0]) * (box[3] - box[1]), 0.0
    )
    area_boxes = np.maximum(
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1]),
        0.0,
    )
    return intersection / np.maximum(
        area_box + area_boxes - intersection, 1e-7
    )


def _average_precision(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(
        np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1])
    )


def _calculate_metrics(detections, ground_truth):
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    detections = sorted(
        detections, key=lambda item: item["score"], reverse=True
    )
    matched = {
        image_id: np.zeros(len(boxes), dtype=bool)
        for image_id, boxes in ground_truth.items()
    }

    tp_flags = []
    matched_ious = []
    for detection in detections:
        image_id = detection["image_id"]
        gt_boxes = ground_truth[image_id]
        ious = _iou_one_to_many(detection["box"], gt_boxes)
        available = np.where(~matched[image_id])[0]
        is_true_positive = False
        matched_iou = 0.0
        if len(available):
            best_local = available[np.argmax(ious[available])]
            matched_iou = float(ious[best_local])
            if matched_iou >= cfg.EVAL_IOU_THRESHOLD:
                matched[image_id][best_local] = True
                is_true_positive = True
        tp_flags.append(is_true_positive)
        matched_ious.append(matched_iou if is_true_positive else 0.0)

    tp = np.asarray(tp_flags, dtype=np.float32)
    fp = 1.0 - tp
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recalls = cumulative_tp / max(total_gt, 1)
    precisions = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1e-12
    )
    ap50 = _average_precision(recalls, precisions) if len(tp) else 0.0

    operating_mask = np.asarray(
        [
            detection["score"] >= cfg.EVAL_SCORE_THRESHOLD
            for detection in detections
        ],
        dtype=bool,
    )
    op_tp = int(np.sum(tp[operating_mask]))
    op_fp = int(np.sum(fp[operating_mask]))
    op_fn = int(total_gt - op_tp)
    precision = op_tp / max(op_tp + op_fp, 1)
    recall = op_tp / max(total_gt, 1)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    true_positive_ious = [
        iou for iou, is_tp in zip(matched_ious, tp_flags) if is_tp
    ]

    return {
        "AP50": ap50,
        "mAP50": ap50,
        "precision_at_0.5": precision,
        "recall_at_0.5": recall,
        "f1_at_0.5": f1,
        "true_positives": op_tp,
        "false_positives": op_fp,
        "false_negatives": op_fn,
        "ground_truth_objects": int(total_gt),
        "detections_for_ap": len(detections),
        "mean_iou_true_positives": (
            float(np.mean(true_positive_ious))
            if true_positive_ious
            else 0.0
        ),
    }


def evaluate_predictor(name, predictor, dataset, verbose=True):
    detections = []
    ground_truth = {}
    image_id = 0
    inference_times = []

    for images, boxes, labels, num_boxes in dataset:
        del labels
        start = time.perf_counter()
        cls_batch, box_batch = predictor(images.numpy())
        inference_times.append(time.perf_counter() - start)

        for batch_index in range(len(images)):
            count = int(num_boxes[batch_index])
            ground_truth[image_id] = (
                boxes[batch_index, :count].numpy().astype(np.float32)
            )
            pred_boxes, pred_scores = _postprocess(
                cls_batch[batch_index],
                box_batch[batch_index],
            )
            for box, score in zip(pred_boxes, pred_scores):
                detections.append(
                    {
                        "image_id": image_id,
                        "score": float(score),
                        "box": box.astype(np.float32),
                    }
                )
            image_id += 1

    metrics = _calculate_metrics(detections, ground_truth)
    metrics["images"] = image_id
    metrics["mean_inference_ms"] = float(
        np.mean(inference_times) * 1000.0
    )
    if verbose:
        print(
            f"  {name:<14} "
            f"mAP50={metrics['mAP50']:.4f}  "
            f"P@0.5={metrics['precision_at_0.5']:.4f}  "
            f"R@0.5={metrics['recall_at_0.5']:.4f}  "
            f"F1={metrics['f1_at_0.5']:.4f}"
        )
    return metrics


def evaluate_keras_model(model, dataset, name="Validation", verbose=False):
    return evaluate_predictor(
        name,
        KerasPredictor(model),
        dataset,
        verbose=verbose,
    )


def evaluate_saved_models(
    output_path=None,
    tfrecord_path=None,
    dataset_title="TEST-SET DETECTION METRICS",
    models_to_evaluate=None,
):
    from models.ssd_mobilenet_v2 import build_teacher
    from models.ssd_mobilenet_v2_lite import build_student

    artifacts = {
        "Teacher": os.path.join(
            cfg.TEACHER_CKPT_DIR, "best_teacher.weights.h5"
        ),
        "Student KD": os.path.join(
            cfg.STUDENT_CKPT_DIR, "best_student.weights.h5"
        ),
        "Pruned": cfg.PRUNED_MODEL_PATH,
        "TFLite INT8": cfg.TFLITE_OUTPUT_PATH,
    }
    if models_to_evaluate is None:
        selected_models = list(artifacts)
    else:
        selected_models = list(models_to_evaluate)
        unknown = [name for name in selected_models if name not in artifacts]
        if unknown:
            raise ValueError(
                "Unknown model selection(s): " + ", ".join(unknown)
            )

    missing = [
        f"{name}: {path}"
        for name, path in artifacts.items()
        if name in selected_models
        if not tf.io.gfile.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Evaluation artifacts are missing:\n  " + "\n  ".join(missing)
        )

    tfrecord_path = tfrecord_path or cfg.TEST_TFRECORD
    dataset = create_detection_eval_dataset(
        tfrecord_path, batch_size=1
    )
    results = {}

    print(f"\n{dataset_title}")
    print(f"  TFRecord: {tfrecord_path}")
    print(
        f"  IoU={cfg.EVAL_IOU_THRESHOLD}, "
        f"precision/recall score threshold="
        f"{cfg.EVAL_SCORE_THRESHOLD}, "
        f"NMS IoU={cfg.NMS_IOU_THRESHOLD}"
    )

    if "Teacher" in selected_models:
        teacher = build_teacher(
            return_features=False, pretrained=False
        )
        teacher.load_weights(artifacts["Teacher"])
        results["teacher"] = evaluate_predictor(
            "Teacher", KerasPredictor(teacher), dataset
        )
        del teacher
        tf.keras.backend.clear_session()

    if "Student KD" in selected_models:
        student = build_student(
            return_features=False, pretrained=False
        )
        student.load_weights(artifacts["Student KD"])
        results["student_kd"] = evaluate_predictor(
            "Student KD", KerasPredictor(student), dataset
        )
        del student
        tf.keras.backend.clear_session()

    if "Pruned" in selected_models:
        pruned = tf.keras.models.load_model(
            artifacts["Pruned"], compile=False
        )
        results["pruned"] = evaluate_predictor(
            "Pruned", KerasPredictor(pruned), dataset
        )
        del pruned
        tf.keras.backend.clear_session()

    if "TFLite INT8" in selected_models:
        results["tflite_int8"] = evaluate_predictor(
            "TFLite INT8",
            TFLitePredictor(artifacts["TFLite INT8"]),
            dataset,
        )

    output_path = output_path or cfg.EVALUATION_OUTPUT_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\n  Saved metrics: {output_path}")
    return results
