"""Three-level knowledge distillation for the MobileNetV2 SSD student."""

import math
import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow.keras import layers

import config as cfg
from models.losses import ssd_loss


class WarmupCosineDecay(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = float(base_lr)
        self.total_steps = float(total_steps)
        self.warmup_steps = float(warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (
            step / tf.maximum(self.warmup_steps, 1.0)
        )
        progress = (
            (step - self.warmup_steps)
            / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        )
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = 0.5 * self.base_lr * (
            1.0 + tf.cos(math.pi * progress)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


def current_learning_rate(optimizer):
    schedule = getattr(optimizer, "_learning_rate", None)
    if callable(schedule):
        return schedule(optimizer.iterations)
    return tf.convert_to_tensor(optimizer.learning_rate)


def _soft_logit_distillation(
    student_logits, teacher_logits, temperature
):
    temperature = tf.cast(temperature, student_logits.dtype)
    teacher_probs = tf.nn.softmax(
        tf.stop_gradient(teacher_logits) / temperature,
        axis=-1,
    )
    student_log_probs = tf.nn.log_softmax(
        student_logits / temperature,
        axis=-1,
    )
    teacher_log_probs = tf.math.log(teacher_probs + 1e-10)
    kl = tf.reduce_sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        axis=-1,
    )
    return tf.reduce_mean(kl) * tf.square(temperature)


def _feature_distillation(
    student_features, teacher_features, adapters
):
    losses = []
    for student_feature, teacher_feature, adapter in zip(
        student_features, teacher_features, adapters
    ):
        adapted = adapter(student_feature)
        target = tf.stop_gradient(teacher_feature)
        losses.append(tf.reduce_mean(tf.square(adapted - target)))
    if not losses:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.add_n(losses) / float(len(losses))


class KDTrainer:
    def __init__(
        self,
        teacher,
        student,
        strategy,
        steps_per_epoch,
        epochs=None,
        base_lr=None,
    ):
        self.teacher = teacher
        self.student = student
        self.strategy = strategy
        self.teacher.trainable = False

        epochs = epochs or cfg.STUDENT_KD_EPOCHS
        base_lr = base_lr or cfg.STUDENT_KD_LR
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * cfg.WARMUP_EPOCHS

        # Multi-GPU requires every variable to exist before strategy.run is
        # traced. Adapters and EMA shadows therefore cannot be created lazily.
        with self.strategy.scope():
            self.lr_schedule = WarmupCosineDecay(
                base_lr, total_steps, warmup_steps
            )
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.lr_schedule
            )

            self.adapters = []
            for index in range(cfg.NUM_DISTILL_FEATURES):
                adapter = layers.Conv2D(
                    cfg.TEACHER_FPN_CHANNELS,
                    1,
                    padding="same",
                    use_bias=False,
                    name=f"kd_adapter_{index}",
                )
                adapter.build(
                    self.student.output[f"feat_{index}"].shape
                )
                self.adapters.append(adapter)

            self.ema_decay = tf.constant(0.999, tf.float32)
            self.ema_weights = [
                tf.Variable(
                    weight.read_value(),
                    trainable=False,
                    name=f"student_ema_{index}",
                )
                for index, weight in enumerate(
                    self.student.trainable_variables
                )
            ]

    def _replica_train_step(self, images, cls_targets, box_targets):
        with tf.GradientTape() as tape:
            teacher_out = self.teacher(images, training=False)
            student_out = self.student(images, training=True)

            hard, cls_loss, loc_loss = ssd_loss(
                student_out["cls_preds"],
                student_out["box_preds"],
                cls_targets,
                box_targets,
            )
            soft = _soft_logit_distillation(
                student_out["cls_preds"],
                teacher_out["cls_preds"],
                cfg.KD_TEMPERATURE,
            )
            feature = _feature_distillation(
                [
                    student_out[f"feat_{i}"]
                    for i in range(cfg.NUM_DISTILL_FEATURES)
                ],
                [
                    teacher_out[f"feat_{i}"]
                    for i in range(cfg.NUM_DISTILL_FEATURES)
                ],
                self.adapters,
            )
            total = (
                cfg.ALPHA_HARD * hard
                + cfg.ALPHA_SOFT * soft
                + cfg.ALPHA_FEATURE * feature
            )
            gradient_loss = (
                total / self.strategy.num_replicas_in_sync
            )

        variables = list(self.student.trainable_variables)
        for adapter in self.adapters:
            variables.extend(adapter.trainable_variables)
        gradients = tape.gradient(gradient_loss, variables)
        self.optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        )
        return {
            "total": total,
            "hard": hard,
            "soft": soft,
            "feature": feature,
            "cls": cls_loss,
            "loc": loc_loss,
        }

    def _update_ema(self):
        for shadow, weight in zip(
            self.ema_weights, self.student.trainable_variables
        ):
            shadow.assign(
                self.ema_decay * shadow
                + (1.0 - self.ema_decay) * weight
            )

    @tf.function
    def distributed_train_step(self, distributed_batch):
        images, cls_targets, box_targets = distributed_batch
        per_replica = self.strategy.run(
            self._replica_train_step,
            args=(images, cls_targets, box_targets),
        )
        reduced = {
            key: self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                value,
                axis=None,
            )
            for key, value in per_replica.items()
        }
        self._update_ema()
        return reduced

    def _replica_validation_step(
        self, images, cls_targets, box_targets
    ):
        outputs = self.student(images, training=False)
        return ssd_loss(
            outputs["cls_preds"],
            outputs["box_preds"],
            cls_targets,
            box_targets,
        )

    @tf.function
    def distributed_validation_step(self, distributed_batch):
        images, cls_targets, box_targets = distributed_batch
        per_replica = self.strategy.run(
            self._replica_validation_step,
            args=(images, cls_targets, box_targets),
        )
        return tuple(
            self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN,
                value,
                axis=None,
            )
            for value in per_replica
        )

    def apply_ema(self):
        for shadow, weight in zip(
            self.ema_weights, self.student.trainable_variables
        ):
            weight.assign(shadow)

    def train(
        self,
        train_dataset,
        val_dataset=None,
        val_metric_dataset=None,
        epochs=None,
        steps_per_epoch=None,
        validation_steps=None,
    ):
        epochs = epochs or cfg.STUDENT_KD_EPOCHS
        if not steps_per_epoch:
            raise ValueError("steps_per_epoch must be a positive integer")
        if val_dataset is None or not validation_steps:
            raise ValueError(
                "val_dataset and validation_steps are required"
            )

        os.makedirs(cfg.STUDENT_CKPT_DIR, exist_ok=True)
        if val_metric_dataset is None:
            from data_loader import create_detection_eval_dataset

            val_metric_dataset = create_detection_eval_dataset(
                cfg.VALID_TFRECORD,
                batch_size=cfg.VAL_METRIC_BATCH_SIZE,
            )

        best_val_map50 = float("-inf")
        epochs_without_improvement = 0
        best_path = os.path.join(
            cfg.STUDENT_CKPT_DIR,
            "best_student.weights.h5",
        )

        for epoch in range(1, epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"  KD Epoch {epoch}/{epochs}")
            print(f"{'=' * 60}")
            totals = {
                key: 0.0
                for key in [
                    "total",
                    "hard",
                    "soft",
                    "feature",
                    "cls",
                    "loc",
                ]
            }

            for step, batch in enumerate(train_dataset):
                if step >= steps_per_epoch:
                    break
                metrics = self.distributed_train_step(batch)
                for key in totals:
                    totals[key] += float(metrics[key])

                if step % 50 == 0:
                    lr = float(current_learning_rate(self.optimizer))
                    print(
                        f"  step {step:>5d}  "
                        f"total={float(metrics['total']):.4f}  "
                        f"hard={float(metrics['hard']):.4f}  "
                        f"soft={float(metrics['soft']):.4f}  "
                        f"feature={float(metrics['feature']):.4f}  "
                        f"lr={lr:.2e}"
                    )

            averages = {
                key: value / steps_per_epoch
                for key, value in totals.items()
            }
            print(
                f"  Epoch {epoch} average: "
                f"total={averages['total']:.4f}  "
                f"hard={averages['hard']:.4f}  "
                f"soft={averages['soft']:.4f}  "
                f"feature={averages['feature']:.4f}"
            )

            live_weights = [
                tf.identity(weight)
                for weight in self.student.trainable_variables
            ]
            self.apply_ema()
            val_totals = [0.0, 0.0, 0.0]
            for val_step, batch in enumerate(val_dataset):
                if val_step >= validation_steps:
                    break
                values = self.distributed_validation_step(batch)
                for index, value in enumerate(values):
                    val_totals[index] += float(value)
            val_loss, val_cls, val_loc = [
                value / validation_steps for value in val_totals
            ]
            print(
                f"  EMA validation: val_loss={val_loss:.4f}  "
                f"val_cls={val_cls:.4f}  val_loc={val_loc:.4f}"
            )
            from evaluation import evaluate_keras_model

            val_metrics = evaluate_keras_model(
                self.student,
                val_metric_dataset,
                name="Validation",
                verbose=False,
            )
            val_map50 = val_metrics["mAP50"]
            print(
                f"  EMA detection validation: "
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
                self.student.save_weights(best_path)
                print(
                    "  Saved best val_mAP50 student weights: "
                    f"{best_path}"
                )
            else:
                epochs_without_improvement += 1
                print(
                    "  No meaningful validation mAP50 improvement "
                    f"({epochs_without_improvement}/"
                    f"{cfg.EARLY_STOPPING_PATIENCE})"
                )

            for saved, weight in zip(
                live_weights, self.student.trainable_variables
            ):
                weight.assign(saved)

            if (
                epochs_without_improvement
                >= cfg.EARLY_STOPPING_PATIENCE
            ):
                print(
                    f"  Early stopping KD at epoch {epoch}; "
                    f"best val_mAP50={best_val_map50:.4f}"
                )
                break

        self.student.load_weights(best_path)
        final_path = os.path.join(
            cfg.STUDENT_CKPT_DIR,
            "final_student_ema.weights.h5",
        )
        self.student.save_weights(final_path)
        print(
            f"  Restored best validation EMA student: {best_path}"
        )
        return self.student
