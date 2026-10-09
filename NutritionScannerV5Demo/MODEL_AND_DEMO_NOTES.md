# NutritionScanner project notes

## Detector packaged in the Android app

The app packages the export originally stored at:

`yolo26_training_results2/yolo26_combined-5/weights/best_int8.tflite`

The checkpoint/export copy is now in the local preparation archive. Its exact
bytes remain included in the repository as
`NutritionScannerV5Demo/app/src/main/assets/nutrition_label_yolo26s_int8.tflite`.
The SHA-256 is
`e3dd5b69c48ad1309a932320f60274e6762de062b16b46deb4702084d98e79df`.
See [the artifact inventory](../docs/ARTIFACTS.md) and
[experimental provenance](../docs/EXPERIMENTS.md).

| Item | Value |
| --- | --- |
| Model | YOLO26s, single class `nutrition_label` |
| App asset | `nutrition_label_yolo26s_int8.tflite` |
| Input | FLOAT32 `[1, 3, 640, 640]`, NCHW RGB, values `[0, 1]` |
| Output | FLOAT32 `[1, 5, 8400]`: normalized `cx, cy, width, height, confidence` |
| Confidence threshold | 0.49 |
| NMS IoU | 0.70 |

The app uses latest-frame CameraX analysis for the live framing outline, then reruns YOLO on the captured photo before it opens the manual crop editor. The model is anchor-free; no legacy MobileNet SSD anchors or offset decoder are used.

## OCR and Gemini flow

1. The participant adjusts the detector suggestion and confirms the crop.
2. Bundled ML Kit OCR reads the selected crop. A local English translation is made when appropriate.
3. The app shows one editable recognized-text field. It uses the English translation when available; the original OCR remains available for reference.
4. Gemini receives the reviewed label text, original OCR when it differs, the OCR language, and the saved health-profile flags. Captured photos remain on-device.
5. Gemini returns plain English guidance. The app extracts normal GenerateContent text and renders the complete reply directly; it does not require a JSON schema, rating, or nutrient-field parser.

The prompt requests 150 to 220 words of general, non-diagnostic nutrition guidance. It notes that OCR order and punctuation can be imperfect, prohibits invented values and medical advice, and requests plain text rather than JSON or Markdown fences.

## Gemini configuration

The research demo reads `GEMINI_API_KEY` from the untracked
`local.properties` file and sends it in the `x-goog-api-key` request
header. Use `local.properties.example` as the setup template.

The app does not log API keys, photos, raw OCR text, or Gemini response bodies. Missing credentials, network failures, rate limits, service failures, safety blocks, and empty responses remain explicit errors that can be retried when appropriate.

## Checks and build

For a fast source and test check without producing an APK:

```powershell
.\gradlew.bat :app:compileDebugKotlin :app:testDebugUnitTest :app:lintDebug
```

When you want to create an installable debug APK:

```powershell
.\gradlew.bat :app:assembleDebug
```
