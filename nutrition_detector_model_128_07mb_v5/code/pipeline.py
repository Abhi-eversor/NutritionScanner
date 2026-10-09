"""End-to-end teacher -> KD -> pruning -> QAT -> INT8 pipeline."""

import argparse
import importlib.util
import math
import os

# TFMOT 0.8 uses the legacy tf_keras API. This must be set before importing
# TensorFlow so every stage creates the same kind of Keras model.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

import config as cfg


def _require_file(path, description):
    if not tf.io.gfile.exists(path):
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )
    return path


def _require_dataset_file(path, description):
    if not tf.io.gfile.exists(path):
        raise FileNotFoundError(
            f"{description} was not found: {path}\n"
            "Set SSD_DATA_DIR to the dataset root containing "
            "train, valid, and test TFRecords."
        )
    return path


def _preflight(step):
    from anchor_utils import NUM_ANCHORS

    if importlib.util.find_spec("tf_keras") is None:
        raise ImportError(
            "tf_keras is required for a pipeline that includes TFMOT QAT. "
            "In the Kaggle setup cell run: "
            "!pip install -q tensorflow-model-optimization tf_keras"
        )
    if step in ("all", "qat") and importlib.util.find_spec(
        "tensorflow_model_optimization"
    ) is None:
        raise ImportError(
            "tensorflow-model-optimization is missing. In Kaggle run: "
            "!pip install -q tensorflow-model-optimization"
        )

    required_files = (
        [(cfg.TEST_TFRECORD, "test TFRecord")]
        if step == "evaluate"
        else [
            (cfg.TRAIN_TFRECORD, "training TFRecord"),
            (cfg.VALID_TFRECORD, "validation TFRecord"),
            (cfg.LABEL_MAP_PATH, "label map"),
        ]
    )
    for path, description in required_files:
        _require_dataset_file(path, description)

    print(f"  TensorFlow: {tf.__version__}")
    print(f"  Code:       {os.path.dirname(os.path.abspath(__file__))}")
    print(f"  Data:       {cfg.DATA_DIR}")
    print(f"  Output:     {cfg.OUTPUT_DIR}")
    print(f"  Image size: {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    print(f"  SSD levels: {cfg.FEATURE_MAP_SIZES}")
    print(f"  Anchors:    {NUM_ANCHORS}")


def _setup_strategy():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    strategy = tf.distribute.MirroredStrategy()
    print(f"  Strategy:   {strategy.__class__.__name__}")
    print(f"  Replicas:   {strategy.num_replicas_in_sync}")
    return strategy


def _make_datasets(strategy):
    from data_loader import (
        count_records,
        create_dataset,
        create_detection_eval_dataset,
    )

    global_batch_size = (
        cfg.BATCH_SIZE_PER_GPU * strategy.num_replicas_in_sync
    )
    train_ds = create_dataset(
        cfg.TRAIN_TFRECORD,
        is_training=True,
        batch_size=global_batch_size,
    )
    val_ds = create_dataset(
        cfg.VALID_TFRECORD,
        is_training=False,
        batch_size=global_batch_size,
    )
    val_metric_ds = create_detection_eval_dataset(
        cfg.VALID_TFRECORD,
        batch_size=cfg.VAL_METRIC_BATCH_SIZE,
    )

    num_train = count_records(cfg.TRAIN_TFRECORD)
    num_val = count_records(cfg.VALID_TFRECORD)
    steps_per_epoch = max(1, num_train // global_batch_size)
    validation_steps = max(1, math.ceil(num_val / global_batch_size))
    print(f"  Training samples:  {num_train}")
    print(f"  Validation samples:{num_val:>6d}")
    print(f"  Global batch size: {global_batch_size}")
    print(f"  Steps per epoch:   {steps_per_epoch}")
    print(f"  Validation steps:  {validation_steps}")

    distributed_train = strategy.experimental_distribute_dataset(
        train_ds
    )
    distributed_val = strategy.experimental_distribute_dataset(val_ds)
    return (
        distributed_train,
        distributed_val,
        val_ds,
        val_metric_ds,
        steps_per_epoch,
        validation_steps,
    )


def _map_improvement(best_map50):
    if math.isfinite(best_map50):
        return max(
            cfg.EARLY_STOPPING_MIN_DELTA,
            abs(best_map50) * cfg.EARLY_STOPPING_MIN_RELATIVE,
        )
    return cfg.EARLY_STOPPING_MIN_DELTA


def _validation_detection_metrics(model, val_metric_dataset):
    from evaluation import evaluate_keras_model

    return evaluate_keras_model(
        model,
        val_metric_dataset,
        name="Validation",
        verbose=False,
    )


def _train_with_tape(
    model,
    dataset,
    val_dataset,
    val_metric_dataset,
    strategy,
    epochs,
    steps_per_epoch,
    validation_steps,
    base_lr,
    checkpoint_dir,
    model_name,
):
    from knowledge_distillation import (
        WarmupCosineDecay,
        current_learning_rate,
    )
    from models.losses import ssd_loss

    with strategy.scope():
        schedule = WarmupCosineDecay(
            base_lr,
            steps_per_epoch * epochs,
            steps_per_epoch * min(cfg.WARMUP_EPOCHS, epochs),
        )
        optimizer = tf.keras.optimizers.Adam(schedule)

    def replica_step(images, cls_targets, box_targets):
        with tf.GradientTape() as tape:
            outputs = model(images, training=True)
            total, cls_loss, loc_loss = ssd_loss(
                outputs["cls_preds"],
                outputs["box_preds"],
                cls_targets,
                box_targets,
            )
            gradient_loss = (
                total / strategy.num_replicas_in_sync
            )
        gradients = tape.gradient(
            gradient_loss, model.trainable_variables
        )
        optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(
                gradients, model.trainable_variables
            )
            if gradient is not None
        )
        return total, cls_loss, loc_loss

    @tf.function
    def distributed_step(batch):
        images, cls_targets, box_targets = batch
        per_replica = strategy.run(
            replica_step,
            args=(images, cls_targets, box_targets),
        )
        return tuple(
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                value,
                axis=None,
            )
            for value in per_replica
        )

    def replica_validation_step(images, cls_targets, box_targets):
        outputs = model(images, training=False)
        return ssd_loss(
            outputs["cls_preds"],
            outputs["box_preds"],
            cls_targets,
            box_targets,
        )

    @tf.function
    def distributed_validation_step(batch):
        images, cls_targets, box_targets = batch
        per_replica = strategy.run(
            replica_validation_step,
            args=(images, cls_targets, box_targets),
        )
        return tuple(
            strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                value,
                axis=None,
            )
            for value in per_replica
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_map50 = float("-inf")
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"best_{model_name}.weights.h5",
    )

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"  [{model_name}] Epoch {epoch}/{epochs}")
        print(f"{'=' * 60}")
        total_loss = 0.0

        for step, batch in enumerate(dataset):
            if step >= steps_per_epoch:
                break
            total, cls_loss, loc_loss = distributed_step(batch)
            total_loss += float(total)
            if step % 50 == 0:
                lr = float(current_learning_rate(optimizer))
                print(
                    f"  step {step:>5d}  loss={float(total):.4f}  "
                    f"cls={float(cls_loss):.4f}  "
                    f"loc={float(loc_loss):.4f}  lr={lr:.2e}"
                )

        train_average = total_loss / steps_per_epoch
        val_totals = [0.0, 0.0, 0.0]
        for val_step, batch in enumerate(val_dataset):
            if val_step >= validation_steps:
                break
            values = distributed_validation_step(batch)
            for index, value in enumerate(values):
                val_totals[index] += float(value)
        val_average, val_cls, val_loc = [
            value / validation_steps for value in val_totals
        ]
        print(
            f"  Epoch {epoch}: train_loss={train_average:.4f}  "
            f"val_loss={val_average:.4f}  "
            f"val_cls={val_cls:.4f}  val_loc={val_loc:.4f}"
        )
        val_metrics = _validation_detection_metrics(
            model,
            val_metric_dataset,
        )
        val_map50 = val_metrics["mAP50"]
        print(
            f"  Detection validation: val_mAP50={val_map50:.4f}  "
            f"P={val_metrics['precision_at_0.5']:.4f}  "
            f"R={val_metrics['recall_at_0.5']:.4f}  "
            f"F1={val_metrics['f1_at_0.5']:.4f}"
        )

        required_improvement = _map_improvement(best_val_map50)
        improved = (
            not math.isfinite(best_val_map50)
            or val_map50 > best_val_map50 + required_improvement
        )
        if improved:
            best_val_map50 = val_map50
            epochs_without_improvement = 0
            model.save_weights(checkpoint_path)
            print(
                f"  Saved best val_mAP50 checkpoint: {checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1
            print(
                "  No meaningful validation mAP50 improvement "
                f"({epochs_without_improvement}/"
                f"{cfg.EARLY_STOPPING_PATIENCE})"
            )
            if (
                epochs_without_improvement
                >= cfg.EARLY_STOPPING_PATIENCE
            ):
                print(
                    f"  Early stopping at epoch {epoch}; "
                    f"best val_mAP50={best_val_map50:.4f}"
                )
                break

    model.load_weights(checkpoint_path)
    return model


def step_train_teacher(
    strategy,
    train_ds,
    val_ds,
    val_metric_ds,
    steps_per_epoch,
    validation_steps,
):
    from models.ssd_mobilenet_v2 import build_teacher

    print("\nSTEP 1 - TRAIN TEACHER")
    with strategy.scope():
        teacher = build_teacher(
            return_features=True,
            pretrained=True,
        )
    print(f"  Teacher parameters: {teacher.count_params():,}")
    return _train_with_tape(
        teacher,
        train_ds,
        val_ds,
        val_metric_ds,
        strategy,
        cfg.TEACHER_EPOCHS,
        steps_per_epoch,
        validation_steps,
        cfg.TEACHER_LR,
        cfg.TEACHER_CKPT_DIR,
        "teacher",
    )


def step_distill(
    strategy,
    train_ds,
    val_ds,
    val_metric_ds,
    steps_per_epoch,
    validation_steps,
    teacher=None,
):
    from knowledge_distillation import KDTrainer
    from models.ssd_mobilenet_v2 import build_teacher
    from models.ssd_mobilenet_v2_lite import build_student

    print("\nSTEP 2 - KNOWLEDGE DISTILLATION")
    with strategy.scope():
        if teacher is None:
            checkpoint = _require_file(
                os.path.join(
                    cfg.TEACHER_CKPT_DIR,
                    "best_teacher.weights.h5",
                ),
                "teacher checkpoint",
            )
            teacher = build_teacher(
                return_features=True,
                pretrained=False,
            )
            teacher.load_weights(checkpoint)
            print(f"  Loaded teacher: {checkpoint}")

        student = build_student(
            return_features=True,
            pretrained=True,
        )

    print(f"  Teacher parameters: {teacher.count_params():,}")
    print(f"  Student parameters: {student.count_params():,}")
    trainer = KDTrainer(
        teacher,
        student,
        strategy,
        steps_per_epoch,
    )
    return trainer.train(
        train_ds,
        val_dataset=val_ds,
        val_metric_dataset=val_metric_ds,
        epochs=cfg.STUDENT_KD_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )


def step_prune(
    strategy,
    train_ds,
    val_ds,
    val_metric_ds,
    steps_per_epoch,
    validation_steps,
    student=None,
):
    from models.ssd_mobilenet_v2_lite import build_student
    from structured_pruning import prune_student

    print("\nSTEP 3 - STRUCTURED PRUNING")
    if student is None:
        checkpoint = os.path.join(
            cfg.STUDENT_CKPT_DIR,
            "best_student.weights.h5",
        )
        if not tf.io.gfile.exists(checkpoint):
            checkpoint = os.path.join(
                cfg.STUDENT_CKPT_DIR,
                "final_student_ema.weights.h5",
            )
        _require_file(checkpoint, "student checkpoint")
        with strategy.scope():
            student = build_student(
                return_features=True,
                pretrained=False,
            )
            student.load_weights(checkpoint)
        print(f"  Loaded student: {checkpoint}")

    with strategy.scope():
        pruned_model, stats = prune_student(
            student,
            return_features=False,
        )
    print(
        f"  Parameter reduction: "
        f"{stats['before_params']:,} -> {stats['after_params']:,}"
    )

    pruned_model = _train_with_tape(
        pruned_model,
        train_ds,
        val_ds,
        val_metric_ds,
        strategy,
        cfg.PRUNE_FINETUNE_EPOCHS,
        steps_per_epoch,
        validation_steps,
        cfg.PRUNE_FINETUNE_LR,
        cfg.PRUNED_CKPT_DIR,
        "pruned_student",
    )
    os.makedirs(cfg.PRUNED_CKPT_DIR, exist_ok=True)
    pruned_model.save(cfg.PRUNED_MODEL_PATH)
    print(f"  Saved pruned model: {cfg.PRUNED_MODEL_PATH}")
    return pruned_model


def step_qat(
    strategy,
    train_ds,
    val_ds,
    val_metric_ds,
    calibration_ds,
    steps_per_epoch,
    validation_steps,
    pruned_model=None,
):
    from quantize_and_convert import quantize_and_export

    print("\nSTEP 4 - QAT AND TFLITE INT8")
    if pruned_model is None:
        _require_file(cfg.PRUNED_MODEL_PATH, "pruned Keras model")
        with strategy.scope():
            pruned_model = tf.keras.models.load_model(
                cfg.PRUNED_MODEL_PATH,
                compile=False,
            )
        print(f"  Loaded pruned model: {cfg.PRUNED_MODEL_PATH}")

    return quantize_and_export(
        pruned_model,
        train_ds,
        val_ds,
        val_metric_ds,
        calibration_ds,
        strategy,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )


def validate_pipeline(strategy, train_ds):
    from models.losses import ssd_loss
    from models.ssd_mobilenet_v2 import build_teacher
    from models.ssd_mobilenet_v2_lite import build_student

    batch = next(iter(train_ds))
    with strategy.scope():
        teacher = build_teacher(
            return_features=True,
            pretrained=False,
        )
        student = build_student(
            return_features=True,
            pretrained=False,
        )

    @tf.function
    def check(distributed_batch):
        def replica_check(images, cls_targets, box_targets):
            outputs = student(images, training=False)
            return ssd_loss(
                outputs["cls_preds"],
                outputs["box_preds"],
                cls_targets,
                box_targets,
            )

        return strategy.run(replica_check, args=distributed_batch)

    losses = check(batch)
    total = strategy.reduce(
        tf.distribute.ReduceOp.MEAN,
        losses[0],
        axis=None,
    )
    print(f"  Teacher parameters: {teacher.count_params():,}")
    print(f"  Student parameters: {student.count_params():,}")
    print(f"  One-batch student loss: {float(total):.4f}")
    print("  Validation passed.")


def print_summary(teacher, student, pruned_model, qat_stats):
    print("\nPIPELINE SUMMARY")
    for name, model in [
        ("Teacher", teacher),
        ("Student KD", student),
        ("Pruned student", pruned_model),
    ]:
        if model is not None:
            print(f"  {name:<16} {model.count_params():>12,} params")
    if qat_stats is not None:
        print(
            f"  TFLite INT8     {qat_stats['size_mb']:.2f} MB, "
            f"{qat_stats['avg_latency_ms']:.2f} ms"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        default="all",
        choices=[
            "all",
            "validate",
            "evaluate",
            "train_teacher",
            "distill",
            "prune",
            "qat",
        ],
    )
    args = parser.parse_args()

    _preflight(args.step)
    if args.step == "evaluate":
        from evaluation import evaluate_saved_models

        evaluate_saved_models()
        return

    strategy = _setup_strategy()
    (
        train_ds,
        val_ds,
        calibration_ds,
        val_metric_ds,
        steps_per_epoch,
        validation_steps,
    ) = _make_datasets(strategy)

    if args.step == "validate":
        validate_pipeline(strategy, train_ds)
        return

    teacher = student = pruned_model = qat_stats = None
    if args.step in ("all", "train_teacher"):
        teacher = step_train_teacher(
            strategy,
            train_ds,
            val_ds,
            val_metric_ds,
            steps_per_epoch,
            validation_steps,
        )
    if args.step in ("all", "distill"):
        student = step_distill(
            strategy,
            train_ds,
            val_ds,
            val_metric_ds,
            steps_per_epoch,
            validation_steps,
            teacher=teacher,
        )
    if args.step in ("all", "prune"):
        pruned_model = step_prune(
            strategy,
            train_ds,
            val_ds,
            val_metric_ds,
            steps_per_epoch,
            validation_steps,
            student=student,
        )
    if args.step in ("all", "qat"):
        _, qat_stats = step_qat(
            strategy,
            train_ds,
            val_ds,
            val_metric_ds,
            calibration_ds,
            steps_per_epoch,
            validation_steps,
            pruned_model=pruned_model,
        )
    if args.step == "all":
        print_summary(
            teacher,
            student,
            pruned_model,
            qat_stats,
        )


if __name__ == "__main__":
    main()
