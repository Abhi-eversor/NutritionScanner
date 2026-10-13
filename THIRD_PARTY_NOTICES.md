# Third-party notices

The root [LICENSE](LICENSE) applies to project-authored code and documentation.
Third-party software, pretrained weights, datasets, and images retain their
own applicable terms. This notice records the dependencies and provenance
that can be established from the retained files; it does not grant rights
that their owners have not granted.

## Software dependencies

Versions and artifact coordinates are recorded in the two applications'
`app/build.gradle.kts` files and `gradle/libs.versions.toml` catalogs. Training
imports and `nutrition_detector_model_128_07mb_v5/code/requirements-kaggle.txt`
record the Python dependencies; they do not establish the complete historical
training environment.

| Component and owner | Use | Upstream license / terms |
| --- | --- | --- |
| AndroidX contributors / Android Open Source Project | Core, Activity, Lifecycle, Compose, Material, CameraX, DataStore, Android tests | [Apache-2.0](https://github.com/androidx/androidx/blob/androidx-main/LICENSE.txt) |
| JetBrains and Kotlin contributors | Kotlin compiler/plugins, standard library | [Apache-2.0](https://github.com/JetBrains/kotlin/blob/master/license/LICENSE.txt) |
| JetBrains and Kotlinx contributors | Coroutines and JSON serialization | [Coroutines: Apache-2.0](https://github.com/Kotlin/kotlinx.coroutines/blob/master/LICENSE.txt); [serialization: Apache-2.0](https://github.com/Kotlin/kotlinx.serialization/blob/master/LICENSE.txt) |
| Square / OkHttp contributors | Gemini REST requests using OkHttp 4.12.0 | [Apache-2.0](https://github.com/square/okhttp/blob/parent-4.12.0/LICENSE.txt) |
| TensorFlow authors | TensorFlow training, TensorFlow Lite Android inference | [Apache-2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE) |
| TF-Keras authors | Legacy Keras model construction and MobileNetV2 backbone | [Apache-2.0](https://github.com/keras-team/tf-keras/blob/master/LICENSE) |
| TensorFlow Model Optimization authors | Quantization-aware training | [Apache-2.0](https://github.com/tensorflow/model-optimization/blob/master/LICENSE) |
| NumPy developers | Training and evaluation array operations | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Gradle contributors | Build system, wrapper scripts and wrapper JARs in both applications | [Apache-2.0](https://github.com/gradle/gradle/blob/master/LICENSE) |
| JUnit contributors | JUnit 4.13.2 tests | [EPL-1.0](https://github.com/junit-team/junit4/blob/r4.13.2/LICENSE-junit.txt) |
| Google ML Kit | OCR recognizers, translation and language identification | [ML Kit terms](https://developers.google.com/ml-kit/terms) and [Google APIs terms](https://developers.google.com/terms), rather than this repository's license |
| Google Gemini API | Optional remote guidance service | [Gemini API additional terms](https://ai.google.dev/gemini-api/terms) and [Google APIs terms](https://developers.google.com/terms) |
| Ultralytics | YOLO26 pretrained detector, training tooling and benchmark export | [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); [upstream model licensing policy](https://www.ultralytics.com/license) |

This table covers direct dependencies identified in the retained source. It
is not a complete inventory of transitive components embedded in an APK.
Redistributions must retain the licenses and notices shipped with their
resolved dependencies, including component-specific notices.
A copy of [Apache-2.0](LICENSE) accompanies the
bundled Gradle wrappers; its upstream source is
[the Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0.txt).

ML Kit libraries and models are obtained through Google's Maven repository
or runtime downloads. The root license does not relicense them. Their
terms include restrictions on reverse engineering and model extraction and
requirements concerning disclosure of ML Kit metrics processing. The
compatibility of a distributed APK containing these components with AGPL
obligations has not been established by this source preparation.

## Bundled models and pretrained backbones

| Artifact | Established provenance |
| --- | --- |
| `app/src/main/assets/nutrition_detector.tflite` and retained MobileNetV2 stage artifacts | Custom MobileNetV2-SSD pipeline in `nutrition_detector_model_128_07mb_v5/code/`, including distillation, pruning and quantization. The model builder requests `tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)` when pretraining is enabled and falls back to random initialization if that download fails. The retained code alone does not prove which initialization succeeded for every historical run. |
| `NutritionScannerV5Demo/app/src/main/assets/nutrition_label_yolo26s_int8.tflite` | YOLO26 benchmark export. Retained YOLO run configurations specify `model: yolo26s.pt` and `pretrained: true`; the exact export command and its association with a particular checkpoint are not recorded. |

MobileNetV2 is credited to Mark Sandler, Andrew Howard, Menglong Zhu,
Andrey Zhmoginov and Liang-Chieh Chen, [MobileNetV2: Inverted Residuals and
Linear Bottlenecks](https://arxiv.org/abs/1801.04381), CVPR 2018. The
[TF-Keras implementation](https://github.com/keras-team/tf-keras/blob/master/tf_keras/applications/mobilenet_v2.py)
credits the TensorFlow authors and documents its ImageNet weight download.
Software licensing is distinct from rights in training images and pretrained
weights. No standalone pretrained-weight grant or complete historical
weight provenance is recorded here. [ImageNet's data access terms](https://www.image-net.org/download.php)
apply to its database; ImageNet images are not included in this repository.

Ultralytics states that its trained and fine-tuned YOLO models are covered
by AGPL-3.0 by default, with an Enterprise license available separately.
No Enterprise license is recorded in this repository. Preserve Ultralytics
attribution and the applicable source obligations when redistributing the
YOLO export. Adopting AGPL for project code does not fill gaps in the
benchmark's training or export provenance.

## Dataset provenance and image rights

The retained training guide identifies these Kaggle dataset locations:

- [Abhi-eversor / test-augment-2 on Kaggle](https://www.kaggle.com/datasets/abhieversor/test-augment-2).
- [Abhi-eversor / validation-data on Kaggle](https://www.kaggle.com/datasets/abhieversor/validation-data).

Original Roboflow export records recovered from the local experiment folder
provide the following project attribution. Both records credit a Roboflow
user without naming an individual creator; the project/workspace references
below preserve the attribution supplied with those exports.

| Recorded export | Source and declared license | Retained evidence |
| --- | --- | --- |
| `test_augment`, version 2; 3,127 images; TFRecord export dated June 17, 2026, 5:38 PM GMT | [experiment-coijn / test_augment](https://universe.roboflow.com/experiment-coijn/test_augment), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [Dataset notice](docs/provenance/test_augment_v2/README.dataset.txt); [export record](docs/provenance/test_augment_v2/README.roboflow.txt) |
| `Nutrition Labels no augs`, version 2; 398 images; TFRecord export dated June 18, 2026, 4:47 PM GMT | [nutrition-yw3l2 / nutrition-labels-no-augs](https://universe.roboflow.com/nutrition-yw3l2/nutrition-labels-no-augs), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [Dataset notice](docs/provenance/nutrition_labels_no_augs_v2/README.dataset.txt); [export record](docs/provenance/nutrition_labels_no_augs_v2/README.roboflow.txt) |

Both exports record auto-orientation, a stretch resize to 640 x 640, and no
image augmentation. These export records identify the recovered local
datasets; their presence and agreement with reported split counts do not
establish that every historical run used those exact files. Kaggle dataset
version IDs and the complete chain of rights in the original images remain
unverified.

There is a license-metadata conflict: the retained
[Kaggle metadata](docs/provenance/test_augment_v2/kaggle-dataset-metadata.json)
identifies `abhieversor/test-augment-2` and declares `CC0-1.0`, whereas its
Roboflow dataset notice declares `CC BY 4.0`. A
[CC0 declaration](https://creativecommons.org/publicdomain/zero/1.0/)
does not waive rights held by other creators. This preparation does not
resolve that conflict or replace the Roboflow attribution requirement with
CC0. Redistributing material under CC BY 4.0 requires appropriate credit,
a license link, and identification of changes.

A version 1 YOLO-format export of `nutrition-labels-no-augs` was also found
locally. This does not identify the constituents of the benchmark's combined
dataset. Its complete source and license attribution remains unresolved.
Raw datasets and prediction contact sheets containing dataset images are
excluded from the upload.

Aggregate metrics and plots without source images are retained as experiment
evidence. Their presence does not establish ownership of the training data
or blanket permission to redistribute model artifacts. Project licensing
grants only the rights held by the project's contributors.
