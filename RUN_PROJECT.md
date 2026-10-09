# Run NutritionScanner

This guide uses repository-relative paths. Run terminal commands from the
repository root, the folder that contains `README.md`, `gradlew`, and
`gradlew.bat`. It works whether the repository was cloned or downloaded into
any local folder.

The root Android project is the primary deployment of the proposed
MobileNetV2-SSD detector. `NutritionScannerV5Demo/` is a separate YOLO
benchmark demonstration and is not the proposed-model application.

## 1. Understand the included artefacts

The repository contains:

- `app/`: primary Android application using MobileNetV2-SSD.
- `nutrition_detector_model_128_07mb_v5/code/`: training, pruning,
  quantization, and evaluation code.
- `nutrition_detector_model_128_07mb_v5/nutrition_detector_run_128_07mb/`:
  MobileNetV2-SSD checkpoints, exported models, and metrics.
- `yolo26_training_results2/`: selected YOLO configurations, result CSVs,
  curves, and confusion matrices retained as research evidence.
- `NutritionScannerV5Demo/`: optional Android demonstration using the YOLO
  benchmark detector.

Keep the published metrics and model artifacts intact when building or running
the apps. Extra YOLO checkpoints and rendered dataset images are excluded and
preserved in a local archive outside the repository; see
[docs/ARTIFACTS.md](docs/ARTIFACTS.md) for the inventory and recovery instructions.
Both Android apps already contain their deployed TFLite assets, so rebuilding
them does not require retraining.

## 2. Install prerequisites

For either Android application, install Android Studio with JDK 17 and Android
SDK Platform 36. Use a physical Android device running Android 8.0 (API 26) or
later with a camera. Both apps require camera hardware, so an emulator without
a camera cannot exercise the complete scan flow.

The detector and bundled OCR recognizers run on-device. Internet access is
needed for Gemini guidance. On-device translation can require a network
connection the first time ML Kit downloads a translation model.

To reproduce training, use Kaggle with a compatible TFRecord dataset. The
reported experiment used two T4 GPUs; see the Kaggle section before changing
the training configuration for different hardware.

## 3. Run the primary MobileNetV2-SSD app

Open the repository root in Android Studio, allow Gradle to sync, choose the
`app` run configuration, connect a device, and run it.

To build from a terminal at the repository root:

```powershell
# Windows PowerShell
.\gradlew.bat :app:assembleDebug
```

```bash
# macOS or Linux
./gradlew :app:assembleDebug
```

### Optional Gemini setup

Gemini guidance is optional. If `local.properties` does not exist, create it
from the template:

```powershell
# Windows PowerShell
Copy-Item .\local.properties.example .\local.properties
```

```bash
# macOS or Linux
cp local.properties.example local.properties
```

If Android Studio has already created `local.properties`, edit that file
instead of copying over it so its existing `sdk.dir` entry remains intact.
Add a valid `GEMINI_API_KEY`. The file is ignored by Git and must not be
committed with a real key. Gradle embeds a configured key in the APK, so do not
publish an APK built with a personal key. Repository validation uses an empty
Gemini key.

### Application flow

1. Grant camera permission.
2. Open the profile screen and save any dietary preference, allergies, and
   optional health flags required for the demonstration.
3. Frame a nutrition panel and capture a photo. Live detection assists
   framing, then MobileNetV2-SSD runs again on the captured image.
4. The app crops the best detected panel, performs on-device OCR, and
   translates to English locally when required.
5. Review and correct the parsed nutrition fields.
6. With a valid API key and network connection, request general nutrition
   guidance. The app sends confirmed fields, saved profile flags, and optional
   reviewed ingredients to Gemini.

Photos and OCR text are not retained as a history. The profile is stored in
local Android DataStore and can be cleared from the profile screen. Android
backup is enabled in the primary app, so profile data may also be included in
system backups according to device settings. Reviewed profile flags are sent
to Gemini when the optional guidance request is made.

## 4. Run the optional YOLO benchmark demo

The optional demo is a separate Android project with the same application ID
as the primary app. Installing one replaces the other on the same device.

From the repository root:

```powershell
# Windows PowerShell
Set-Location .\NutritionScannerV5Demo
.\gradlew.bat :app:assembleDebug
```

```bash
# macOS or Linux
cd NutritionScannerV5Demo
./gradlew :app:assembleDebug
```

Open `NutritionScannerV5Demo/` as a separate Android Studio project when
using the IDE. Its Gemini configuration is independent and uses its own
`local.properties.example` file.

The benchmark demo follows a different interaction flow:

1. Grant camera permission and frame the label using the live detection
   outline.
2. Capture the image so the detector can run on the full photo.
3. Adjust and confirm the proposed crop.
4. Review the OCR text and English translation, when available.
5. Request general guidance only after reviewing text and profile values.

Use this app when demonstrating the benchmark detector or manual crop
confirmation. Use the root Android app when demonstrating the proposed
MobileNetV2-SSD deployment.

## 5. Reproduce the MobileNetV2-SSD pipeline on Kaggle

The raw training dataset is not included in the repository. Attach a
compatible TFRecord export to a Kaggle notebook. The expected layout is:

```text
dataset-root/
  train/test_augment.tfrecord
  train/test_augment_label_map.pbtxt
  valid/test_augment.tfrecord
  test/test_augment.tfrecord
```

Attach the contents of `nutrition_detector_model_128_07mb_v5/code/` as a
Kaggle input dataset. In a fresh notebook, set `CODE` and `DATA` to the
paths Kaggle assigns to those two attached datasets:

```python
CODE = "/kaggle/input/your-training-code"
DATA = "/kaggle/input/your-dataset"
```

Install the QAT dependencies and specify the dataset root:

```python
!pip install -q -r "$CODE/requirements-kaggle.txt"
%env SSD_DATA_DIR=$DATA
```

The reported experiment used two T4 GPUs and a per-GPU batch of 64, producing
a global batch of 128. The pipeline calculates its runtime training batch from
the GPUs visible to TensorFlow, so a one-GPU run uses a global batch of 64 by
default. Reduce `BATCH_SIZE_PER_GPU` if memory requires it. `NUM_GPUS`
defines the configured `GLOBAL_BATCH_SIZE` for code paths that use that
constant directly; set it to 1 when using those helpers independently.

A one-GPU run is useful for functional reproduction, but its hardware and
batch behavior differ from the reported two-T4 experiment. Its training
metrics are therefore not directly comparable to the reported results.

Run the preflight check:

```python
!python "$CODE/pipeline.py" --step validate
```

Run the complete sequence:

```python
!python "$CODE/pipeline.py" --step all
```

The pipeline trains the teacher, distils the smaller student, applies
structured pruning and fine-tuning, then performs quantization-aware training
and exports the final INT8 TFLite model.

By default, artefacts are written to `/kaggle/working`:

- `checkpoints/teacher/best_teacher.weights.h5`
- `checkpoints/student_kd/best_student.weights.h5`
- `checkpoints/pruned/pruned_student.keras`
- `checkpoints/qat/best_qat.weights.h5`
- `nutrition_detector.tflite`
- `test_metrics.json`

To resume after an interruption, run a single stage only after its prerequisite
artefact is available:

```python
!python "$CODE/pipeline.py" --step train_teacher
!python "$CODE/pipeline.py" --step distill
!python "$CODE/pipeline.py" --step prune
!python "$CODE/pipeline.py" --step qat
```

Evaluate the saved models on the configured test split:

```python
!python "$CODE/pipeline.py" --step evaluate
```

This writes `/kaggle/working/test_metrics.json` with AP50/mAP50, precision,
recall, F1, TP, FP, and FN for the teacher, KD student, pruned student, and
final INT8 TFLite model.

To evaluate an external single-class Roboflow TFRecord export:

```python
EXTERNAL_DATA = "/kaggle/input/validation-data"

!python "$CODE/evaluate_external_dataset.py" \
    --data-dir "$EXTERNAL_DATA" \
    --split test \
    --class-name "nutrition table" \
    --models all
```

The external evaluator maps every valid box to foreground class 1. Use
`--models final` when only the final INT8 model needs evaluation. See
`nutrition_detector_model_128_07mb_v5/code/KAGGLE_RUN.md` for a focused
Kaggle guide. Read [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the known
dataset references, configuration, evidence gaps, and threshold-selection
limitations. Recovered Roboflow version-2 export records are included under
`docs/provenance/`; exact historical input identity and the original training
environment still require confirmation.

## 6. Inspect existing research evidence

The MobileNetV2-SSD experiment artefacts are under:

```text
nutrition_detector_model_128_07mb_v5/nutrition_detector_run_128_07mb/
```

The YOLO benchmark outputs are under:

```text
yolo26_training_results2/
```

Configurations, CSVs, and analytical plots are retained for traceability.
Bulk prediction images, image montages, and additional YOLO weights are in the
local preparation archive rather than Git. The benchmark export used by the
app remains in `NutritionScannerV5Demo/app/src/main/assets/`.
See [docs/ARTIFACTS.md](docs/ARTIFACTS.md) for hashes and the archive manifest.

## 7. Run fast Android checks

From the repository root, run:

```powershell
# Windows PowerShell
.\gradlew.bat :app:compileDebugKotlin :app:testDebugUnitTest :app:lintDebug
```

```bash
# macOS or Linux
./gradlew :app:compileDebugKotlin :app:testDebugUnitTest :app:lintDebug
```

Repeat these checks inside `NutritionScannerV5Demo/` to verify the independent
benchmark project. The primary app currently has no authored unit tests; the
benchmark app has YOLO-coordinate and Gemini-parser tests. A successful
`testDebugUnitTest` task for the primary app is not evidence of behavioral test
coverage. See [docs/PREPARATION_REPORT.md](docs/PREPARATION_REPORT.md) for the
recorded build and review results.

If Gemini is not configured, detection and OCR can still be tested. Add a
valid key only to the appropriate project-local `local.properties` file.

If Kaggle cannot find the dataset, verify the TFRecord layout and set
`SSD_DATA_DIR` to the dataset root before running `pipeline.py`. If a
resumed stage fails, verify the preceding checkpoint under
`/kaggle/working/checkpoints/`.
