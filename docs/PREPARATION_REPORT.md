# Initial upload preparation

Prepared on **2026-10-10**. This report records source-repository preparation
and build checks; it does not report a new training or evaluation run.

## Retention and cleanup

- Preserved both Android applications, their runtime models, complete Gradle
  wrappers, and the MobileNet training/compression/evaluation/deployment code.
- Retained all MobileNet stage checkpoints and the original supplied metric
  JSONs, plus selected YOLO configurations, CSVs, and analytical plots.
- Archived 825 YOLO checkpoint/export copies, bulk prediction images, and
  dataset-image montages: 648,416,277 bytes (618.378 MiB). Every moved file's
  SHA-256 was verified after moving. The archive is a sibling directory,
  `NutritionScanner-preparation-archive-2026-10-10`, outside this Git root.
- Verified the two bundled runtime models against their original exports.
  All 51 retained model/report entries match the staged Git blobs exactly.
  See [ARTIFACTS.md](ARTIFACTS.md) and its manifests.
- Added ignore rules for generated outputs, machine-local configuration,
  credentials/signing material, APKs, and excluded YOLO artifacts. Unix
  wrappers are staged executable; Git attributes preserve research reports'
  exact bytes and wrapper line endings across platforms.

## Original research workspace investigation

The separate `VCR_Lab` workspace was inspected read-only: loose source,
metrics, notebooks, export metadata, source archives, and available model
artifacts. No notebooks, original training code, or original evaluation code
were executed. Its `.git` directory is empty and supplies no history.

Both original external-metric JSON copies report F1 0.6315789474. The original
v5 checkpoints, deployed TFLite file, and external metrics match the upload
snapshot byte-for-byte. The original v5 pipeline `test_metrics.json` has a
leading `67` before its opening brace and is invalid JSON; after that prefix,
its bytes are exactly the same as the existing valid upload snapshot. Neither
file was modified during this preparation.

No report supporting historical F1 0.6966 or the claimed confidence-0.30
sweep result 0.7312 was located. These claims remain preserved and labeled
unverified. The evaluator's fixed `*_at_0.5` field names also fail to record
threshold overrides, so the historical operating point cannot be authenticated
from the saved keys alone. See [EXPERIMENTS.md](EXPERIMENTS.md).

Recovered Roboflow version-2 records identify candidate training/external
sources. Read-only TFRecord framing counts match the saved test sizes, but
matching counts alone do not establish exact historical dataset identity or
independent splits. Metadata copies, checksums, and scan limits are under
[provenance/](provenance/README.md). The separate recovered 256-pixel experiment
was not used to fill gaps in the deployed 128-pixel model's results or
environment. The historical training environment remains unverified.

## Android checks

Environment: Windows PowerShell, JDK 17.0.12, Gradle 9.1.0, Android Gradle
Plugin 9.0.1, Kotlin 2.3.20, Android SDK Platform 36 revision 2 (extension 17),
and Build Tools 36.0.0. Both projects were
built independently with empty Gemini keys and their packaged models.

From each Android project root:

```powershell
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-17'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
.\gradlew.bat :app:assembleDebug :app:testDebugUnitTest :app:lintDebug --no-daemon --console=plain
```

| Project | Debug APK | Existing unit tests | Lint |
| --- | --- | --- | --- |
| Root MobileNet app | Passed | `NO-SOURCE`; no authored unit tests | Passed: 0 errors, 42 warnings |
| `NutritionScannerV5Demo/` | Passed | 9 passed; 0 failures/errors/skips | Passed: 0 errors, 43 warnings |

The first root build exposed invalid/obsolete imports. The only Android
source fixes removed two CameraX `toBitmap` extension imports and Compose
scope-function imports, corrected `ContentScale`'s package, and imported the
JSON-array `add` overload. No dependency version, detector behavior, model
bytes, or experimental result changed. The root rerun passed. The benchmark
app required no source edits.

Lint and compiler warnings remain recorded rather than treated as passing
runtime tests. Root lint includes native-library 16 KB page-alignment
warnings (six in the primary app, three in the benchmark). Physical-device
scanning, OCR/translation downloads, Gemini
responses, and compatibility across Android devices were not tested. Training
Python syntax was checked for all 13 `.py` files without importing TensorFlow
or executing training.

Build logs and detailed reports are stored locally outside the repository;
the original console logs are under
`%TEMP%\NutritionScanner-upload-builds-20261010\`. Generated APKs and
`BuildConfig` files are ignored and both generated Gemini keys were confirmed
empty.
A copy of the console logs, lint reports, and benchmark test XMLs is preserved
under `build-evidence/` in the sibling preparation archive. Both final Gradle
invocations exited successfully; the root retry took 3 minutes 58 seconds and
the benchmark took 3 minutes 5 seconds.

## Credential and licensing review

The candidate upload was scanned for common Google/GitHub/AWS/Slack credential
formats, private-key markers, JWTs, nonempty credential-like assignments, and
sensitive filenames. No matching secrets or sensitive configuration files
were found in the upload. Both local-properties examples are blank templates.
This is a pattern-based scan, not a guarantee about all possible secret formats.

`VCR_Lab/test.ipynb` contains hardcoded Roboflow API-key arguments. That
notebook remains outside the upload; key values were not printed or copied.
No original credentials or research files were changed.

Added AGPL-3.0-only for project-authored source, third-party attribution, and
an Apache-2.0 license copy for the bundled Gradle wrappers. Recovered export
notices declare CC BY 4.0, while the training dataset's Kaggle metadata lists
CC0. This conflict and unverified source-image lineage are explicit in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); the project license does
not relicense third-party libraries, datasets, or model artifacts.

## Staged-file review

The import includes source, build tooling, documentation, model assets,
MobileNet checkpoints, and selected research reports: 185 files, approximately
55.9 MiB of staged contents. It excludes Android
caches/build outputs, keyed APKs, notebooks containing credentials, raw
datasets, source archives, and all 825 archived paths. No file exceeds 50 MiB.

Final review checks include staged credential scanning, artifact byte/hash
comparison, ignore-rule checks, Markdown link/fence checks, and
`git diff --cached --check`. The initial commit is local; no repository was
created on GitHub and no files were pushed.
