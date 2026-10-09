"""Configuration for the MobileNetV2 SSD training pipeline."""

import glob
import os

IMAGE_SIZE = 128                          # MCU-lite training & inference resolution
NUM_CLASSES = 2                           # 0 = background, 1 = nutrition_facts

TEACHER_ALPHA = 1.0
TEACHER_FPN_CHANNELS = 512

STUDENT_ALPHA = 0.35
STUDENT_FPN_CHANNELS = 64

BACKBONE_C4_LAYER = "block_12_project_BN"
BACKBONE_C5_LAYER = "out_relu"

FEATURE_MAP_SIZES = [8, 4, 2, 1]
NUM_FEATURE_MAPS = len(FEATURE_MAP_SIZES)

ASPECT_RATIOS = [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]
NUM_ANCHORS_PER_LOC = len(ASPECT_RATIOS) + 1

ANCHOR_SCALE_MIN = 0.2
ANCHOR_SCALE_MAX = 0.9

IOU_POS_THRESHOLD = 0.5
IOU_NEG_THRESHOLD = 0.3

BOX_VARIANCE = [0.1, 0.1, 0.2, 0.2]

BATCH_SIZE_PER_GPU = 64
NUM_GPUS = 2
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS

TEACHER_LR = 1.5e-3
STUDENT_KD_LR = 1.5e-3
PRUNE_FINETUNE_LR = 1e-4
QAT_LR = 5e-5

TEACHER_EPOCHS = 320
STUDENT_KD_EPOCHS = 240
PRUNE_FINETUNE_EPOCHS = 60
QAT_EPOCHS = 40

WARMUP_EPOCHS = 20
WARMUP_LR_FACTOR = 0.1

EARLY_STOPPING_PATIENCE = 32
EARLY_STOPPING_MIN_DELTA = 1e-5
EARLY_STOPPING_MIN_RELATIVE = 1e-5
VAL_METRIC_BATCH_SIZE = 32

RANDOM_CROP_PROB = 0.25
RANDOM_HORIZONTAL_FLIP_PROB = 0.5
RANDOM_COLOR_JITTER_PROB = 0.5

KD_TEMPERATURE = 4.0
ALPHA_HARD = 1.0
ALPHA_SOFT = 1.0
ALPHA_FEATURE = 0.5
NUM_DISTILL_FEATURES = 3

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
LOC_LOSS_WEIGHT = 1.0

EXPANSION_PRUNE_RATIO = 0.10
MIN_EXPANSION_CHANNELS = 8

QAT_NUM_CALIBRATION_IMAGES = 100

NMS_SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS_PER_IMAGE = 20

EVAL_IOU_THRESHOLD = 0.5
EVAL_SCORE_THRESHOLD = 0.5
EVAL_AP_MIN_SCORE = 0.001
EVAL_MAX_DETECTIONS = 100

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_data_dir():
    configured_data_dir = os.environ.get("SSD_DATA_DIR")
    if configured_data_dir:
        return os.path.abspath(configured_data_dir)

    candidates = [
        "/kaggle/input/test-augment-2",
        os.path.join(_CODE_DIR, "test_augment-2"),
        os.path.join(os.getcwd(), "test_augment-2"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(
            os.path.join(candidate, "train", "test_augment.tfrecord")
        ):
            return os.path.abspath(candidate)

    matches = glob.glob(
        "/kaggle/input/**/train/test_augment.tfrecord", recursive=True
    )
    if matches:
        return os.path.dirname(os.path.dirname(matches[0]))

    return "/kaggle/input/test-augment-2"


DATA_DIR = _find_data_dir()

TRAIN_TFRECORD = os.path.join(DATA_DIR, "train", "test_augment.tfrecord")
VALID_TFRECORD = os.path.join(DATA_DIR, "valid", "test_augment.tfrecord")
TEST_TFRECORD  = os.path.join(DATA_DIR, "test",  "test_augment.tfrecord")
LABEL_MAP_PATH = os.path.join(DATA_DIR, "train", "test_augment_label_map.pbtxt")

OUTPUT_DIR = os.environ.get(
    "SSD_OUTPUT_DIR",
    "/kaggle/working" if os.path.isdir("/kaggle/working")
    else os.path.join(_CODE_DIR, "output"),
)
TEACHER_CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints", "teacher")
STUDENT_CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints", "student_kd")
PRUNED_CKPT_DIR    = os.path.join(OUTPUT_DIR, "checkpoints", "pruned")
QAT_CKPT_DIR       = os.path.join(OUTPUT_DIR, "checkpoints", "qat")
PRUNED_MODEL_PATH  = os.path.join(PRUNED_CKPT_DIR, "pruned_student.keras")
TFLITE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nutrition_detector.tflite")
EVALUATION_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "test_metrics.json")
SAVED_MODEL_DIR    = os.path.join(OUTPUT_DIR, "saved_model")
LOG_DIR            = os.path.join(OUTPUT_DIR, "logs")
