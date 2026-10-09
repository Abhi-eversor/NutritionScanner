"""Evaluate saved detector artifacts on an external Roboflow TFRecord dataset."""

import argparse
import glob
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

import config as cfg
from data_loader import count_records
from evaluation import evaluate_saved_models


MODEL_CHOICES = {
    "all": ["Teacher", "Student KD", "Pruned", "TFLite INT8"],
    "teacher": ["Teacher"],
    "student": ["Student KD"],
    "kd": ["Student KD"],
    "pruned": ["Pruned"],
    "final": ["TFLite INT8"],
    "int8": ["TFLite INT8"],
    "tflite": ["TFLite INT8"],
}


def _unique_sorted(paths):
    seen = set()
    unique = []
    for path in paths:
        absolute = os.path.abspath(path)
        if absolute not in seen:
            unique.append(absolute)
            seen.add(absolute)
    return sorted(unique)


def _path_contains_split(path, split):
    parts = path.replace("\\", "/").lower().split("/")
    return split.lower() in parts


def find_tfrecord(data_dir, split):
    if not data_dir:
        raise ValueError("Either --data-dir or --test-tfrecord must be set")

    data_dir = os.path.abspath(data_dir)
    split_dir = os.path.join(data_dir, split)
    split_matches = []
    if os.path.isdir(split_dir):
        split_matches = glob.glob(os.path.join(split_dir, "*.tfrecord"))
    if split_matches:
        return _unique_sorted(split_matches)[0]

    recursive_matches = glob.glob(
        os.path.join(data_dir, "**", "*.tfrecord"),
        recursive=True,
    )
    recursive_matches = _unique_sorted(recursive_matches)
    split_matches = [
        path for path in recursive_matches
        if _path_contains_split(path, split)
    ]
    if split_matches:
        return split_matches[0]
    if recursive_matches:
        print(
            f"Warning: no '{split}' split TFRecord found; using "
            f"{recursive_matches[0]}"
        )
        return recursive_matches[0]

    raise FileNotFoundError(
        f"No .tfrecord files found under dataset folder: {data_dir}"
    )


def find_label_map_for_tfrecord(tfrecord_path):
    folder = os.path.dirname(os.path.abspath(tfrecord_path))
    matches = glob.glob(os.path.join(folder, "*label_map.pbtxt"))
    return _unique_sorted(matches)[0] if matches else None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved nutrition-label detector artifacts on an "
            "external single-class Roboflow TFRecord dataset."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("EXTERNAL_DATA_DIR"),
        help=(
            "Root dataset folder containing train/valid/test subfolders. "
            "Example: /kaggle/input/validation-data"
        ),
    )
    parser.add_argument(
        "--test-tfrecord",
        default=os.environ.get("EXTERNAL_TEST_TFRECORD"),
        help=(
            "Direct path to a TFRecord. Overrides --data-dir/--split. "
            "Example: /kaggle/input/validation-data/test/"
            "stickers-and-tables.tfrecord"
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate when --data-dir is used.",
    )
    parser.add_argument(
        "--class-name",
        default="nutrition table",
        help="Human-readable external dataset class name for reporting.",
    )
    parser.add_argument(
        "--models",
        default="final",
        choices=sorted(MODEL_CHOICES),
        help=(
            "Which saved artifact(s) to evaluate. Use 'final' for the INT8 "
            "TFLite model, or 'all' to compare teacher/KD/pruned/INT8."
        ),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=cfg.EVAL_SCORE_THRESHOLD,
        help="Confidence threshold used for printed precision/recall/F1.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=cfg.NMS_IOU_THRESHOLD,
        help="NMS IoU threshold used during evaluation.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(cfg.OUTPUT_DIR, "external_test_metrics.json"),
        help="Where to write the metrics JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg.EVAL_SCORE_THRESHOLD = args.score_threshold
    cfg.NMS_IOU_THRESHOLD = args.nms_iou

    tfrecord_path = (
        os.path.abspath(args.test_tfrecord)
        if args.test_tfrecord
        else find_tfrecord(args.data_dir, args.split)
    )
    if not tf.io.gfile.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord not found: {tfrecord_path}")

    label_map_path = find_label_map_for_tfrecord(tfrecord_path)
    num_records = count_records(tfrecord_path)

    print("\nEXTERNAL DATASET")
    print(f"  Class name: {args.class_name}")
    print("  Single-class mapping: all valid boxes -> foreground class 1")
    print(f"  TFRecord:   {tfrecord_path}")
    print(f"  Records:    {num_records}")
    if label_map_path:
        print(f"  Label map:  {label_map_path}")

    evaluate_saved_models(
        output_path=args.output,
        tfrecord_path=tfrecord_path,
        dataset_title=f"EXTERNAL METRICS: {args.class_name}",
        models_to_evaluate=MODEL_CHOICES[args.models],
    )


if __name__ == "__main__":
    main()
