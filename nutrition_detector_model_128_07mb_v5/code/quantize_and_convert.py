"""Quantization-aware fine-tuning and full-INT8 TFLite export."""

import math
import os
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

import config as cfg
from knowledge_distillation import (
    WarmupCosineDecay,
    current_learning_rate,
)
from models.losses import ssd_loss


def apply_qat(model):
    import tensorflow_model_optimization as tfmot

    if not type(model).__module__.startswith("tf_keras"):
        raise TypeError(
            "TFMOT requires a tf_keras Functional model. Run pipeline.py "
            "in a fresh Python process so TF_USE_LEGACY_KERAS=1 is applied "
            "before TensorFlow is imported."
        )
    qat_model = tfmot.quantization.keras.quantize_model(model)
    print(
        f"  QAT wrappers applied: {model.count_params():,} -> "
        f"{qat_model.count_params():,} parameters"
    )
    return qat_model


def qat_fine_tune(
    qat_model,
    train_dataset,
    val_dataset,
    val_metric_dataset,
    strategy,
    epochs=None,
    steps_per_epoch=None,
    validation_steps=None,
    base_lr=None,
):
    epochs = epochs or cfg.QAT_EPOCHS
    base_lr = base_lr or cfg.QAT_LR
    if not steps_per_epoch:
        raise ValueError("steps_per_epoch must be a positive integer")
    if val_dataset is None or not validation_steps:
        raise ValueError(
            "val_dataset and validation_steps are required"
        )

    with strategy.scope():
        schedule = WarmupCosineDecay(
            base_lr,
            steps_per_epoch * epochs,
            steps_per_epoch * min(2, epochs),
        )
        optimizer = tf.keras.optimizers.Adam(schedule)

    def replica_step(images, cls_targets, box_targets):
        with tf.GradientTape() as tape:
            outputs = qat_model(images, training=True)
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
            gradient_loss, qat_model.trainable_variables
        )
        optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(
                gradients, qat_model.trainable_variables
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
        outputs = qat_model(images, training=False)
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

    os.makedirs(cfg.QAT_CKPT_DIR, exist_ok=True)
    checkpoint = os.path.join(
        cfg.QAT_CKPT_DIR, "best_qat.weights.h5"
    )
    best_val_map50 = float("-inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        print(f"\n  QAT Epoch {epoch}/{epochs}")
        for step, batch in enumerate(train_dataset):
            if step >= steps_per_epoch:
                break
            total, cls_loss, loc_loss = distributed_step(batch)
            total_loss += float(total)
            if step % 50 == 0:
                lr = float(current_learning_rate(optimizer))
                print(
                    f"    step {step:>5d}  "
                    f"loss={float(total):.4f}  "
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
            f"  QAT epoch: train_loss={train_average:.4f}  "
            f"val_loss={val_average:.4f}  "
            f"val_cls={val_cls:.4f}  val_loc={val_loc:.4f}"
        )
        from evaluation import evaluate_keras_model

        val_metrics = evaluate_keras_model(
            qat_model,
            val_metric_dataset,
            name="Validation",
            verbose=False,
        )
        val_map50 = val_metrics["mAP50"]
        print(
            f"  QAT detection validation: "
            f"val_mAP50={val_map50:.4f}  "
            f"P={val_metrics['precision_at_0.5']:.4f}  "
            f"R={val_metrics['recall_at_0.5']:.4f}  "
            f"F1={val_metrics['f1_at_0.5']:.4f}"
        )

        required_improvement = (
            max(
                cfg.EARLY_STOPPING_MIN_DELTA,
                abs(best_val_map50)
                * cfg.EARLY_STOPPING_MIN_RELATIVE,
            )
            if math.isfinite(best_val_map50)
            else cfg.EARLY_STOPPING_MIN_DELTA
        )
        improved = (
            not math.isfinite(best_val_map50)
            or val_map50 > best_val_map50 + required_improvement
        )
        if improved:
            best_val_map50 = val_map50
            epochs_without_improvement = 0
            qat_model.save_weights(checkpoint)
            print(
                f"  Saved best val_mAP50 QAT checkpoint: {checkpoint}"
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
                    f"  Early stopping QAT at epoch {epoch}; "
                    f"best val_mAP50={best_val_map50:.4f}"
                )
                break

    qat_model.load_weights(checkpoint)
    return qat_model


def representative_dataset_gen(dataset, num_samples=None):
    num_samples = num_samples or cfg.QAT_NUM_CALIBRATION_IMAGES
    count = 0
    for images, _, _ in dataset:
        for image in images:
            yield [image[tf.newaxis, ...].numpy().astype(np.float32)]
            count += 1
            if count >= num_samples:
                return


def _inference_only_model(qat_model):
    outputs = qat_model.output
    if isinstance(outputs, dict):
        selected = [outputs["cls_preds"], outputs["box_preds"]]
    else:
        output_map = dict(zip(qat_model.output_names, qat_model.outputs))
        selected = [
            output_map["cls_preds"],
            output_map["box_preds"],
        ]
    return tf.keras.Model(
        qat_model.inputs,
        selected,
        name="nutrition_detector_int8_export",
    )


def convert_to_tflite(
    qat_model,
    calibration_dataset,
    output_path=None,
):
    output_path = output_path or cfg.TFLITE_OUTPUT_PATH
    inference_model = _inference_only_model(qat_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(
        inference_model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        representative_dataset_gen(calibration_dataset)
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print("  Converting QAT model to TFLite INT8...")
    model_bytes = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as file:
        file.write(model_bytes)
    print(
        f"  Saved TFLite model: {output_path} "
        f"({len(model_bytes) / 1024 ** 2:.2f} MB)"
    )
    return output_path, len(model_bytes)


def benchmark_tflite(tflite_path, num_runs=30):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    shape = input_info["shape"]
    dtype = input_info["dtype"]

    if dtype == np.uint8:
        sample = np.random.randint(0, 256, shape, dtype=np.uint8)
    elif dtype == np.int8:
        sample = np.random.randint(-128, 128, shape, dtype=np.int8)
    else:
        sample = np.random.uniform(-1.0, 1.0, shape).astype(dtype)

    for _ in range(5):
        interpreter.set_tensor(input_info["index"], sample)
        interpreter.invoke()

    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_info["index"], sample)
        interpreter.invoke()
        timings.append(time.perf_counter() - start)

    average_ms = float(np.mean(timings) * 1000.0)
    print(f"  TFLite CPU latency: {average_ms:.2f} ms")
    return average_ms


def quantize_and_export(
    pruned_model,
    train_dataset,
    val_dataset,
    val_metric_dataset,
    calibration_dataset,
    strategy,
    steps_per_epoch=None,
    validation_steps=None,
):
    with strategy.scope():
        qat_model = apply_qat(pruned_model)
    qat_model = qat_fine_tune(
        qat_model,
        train_dataset,
        val_dataset,
        val_metric_dataset,
        strategy,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    path, size_bytes = convert_to_tflite(
        qat_model, calibration_dataset
    )
    latency_ms = benchmark_tflite(path)
    return path, {
        "tflite_path": path,
        "size_bytes": size_bytes,
        "size_mb": size_bytes / 1024 ** 2,
        "avg_latency_ms": latency_ms,
    }
