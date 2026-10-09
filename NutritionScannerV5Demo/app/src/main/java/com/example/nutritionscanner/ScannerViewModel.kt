package com.example.nutritionscanner

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.nutritionscanner.detection.Detection
import com.example.nutritionscanner.detection.NutritionDetectorModel
import com.example.nutritionscanner.llm.GeminiClient
import com.example.nutritionscanner.llm.GeminiAdvice
import com.example.nutritionscanner.llm.GeminiResult
import com.example.nutritionscanner.ocr.NutritionOcrResult
import com.example.nutritionscanner.ocr.OcrProcessor
import com.example.nutritionscanner.profile.HealthProfile
import com.example.nutritionscanner.profile.HealthProfileRepository
import com.example.nutritionscanner.translation.TranslationManager
import com.example.nutritionscanner.translation.TranslationResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** A completed OCR/translation pass waiting for the participant to review text. */
data class NutritionReview(
    val croppedBitmap: Bitmap,
    val ocrResult: NutritionOcrResult,
    val translationResult: TranslationResult,
    val usedFullImageFallback: Boolean,
)

/** Exact reviewed text and profile context retained for an explicit retry. */
data class LabelAnalysisRequest(
    val review: NutritionReview,
    val reviewedLabelText: String,
    val originalOcrText: String?,
    val sourceLanguage: String,
    val profile: HealthProfile,
)

/** A captured photo plus the detector's suggested, user-editable crop. */
data class CropReview(
    val capturedBitmap: Bitmap,
    val initialCrop: RectF,
    val detectorFound: Boolean,
)

/** UI states for capture, confirmation, profile editing, and recommendations. */
sealed class ScanState {
    data object Scanning : ScanState()
    data class Capturing(val capturedBitmap: Bitmap) : ScanState()
    data class AdjustingCrop(val cropReview: CropReview) : ScanState()
    data class ExtractingText(val croppedBitmap: Bitmap, val usedFullImageFallback: Boolean) : ScanState()
    data class Translating(
        val croppedBitmap: Bitmap,
        val ocrResult: NutritionOcrResult,
        val usedFullImageFallback: Boolean,
    ) : ScanState()
    data class ReviewNutrition(val review: NutritionReview) : ScanState()
    data class Analyzing(val request: LabelAnalysisRequest) : ScanState()
    data class Results(
        val request: LabelAnalysisRequest,
        val advice: GeminiAdvice,
    ) : ScanState()
    data class ProfileEditor(val profile: HealthProfile) : ScanState()
    data class Error(
        val message: String,
        val retryable: Boolean = false,
        val retryRequest: LabelAnalysisRequest? = null,
    ) : ScanState()
}

class ScannerViewModel(application: Application) : AndroidViewModel(application) {

    private val detector = NutritionDetectorModel(application)
    private val ocrProcessor = OcrProcessor()
    private val translationManager = TranslationManager()
    private val geminiClient = GeminiClient()
    private val profileRepository = HealthProfileRepository(application)

    private val _detections = MutableStateFlow<List<Detection>>(emptyList())
    val detections: StateFlow<List<Detection>> = _detections.asStateFlow()

    private val _scanState = MutableStateFlow<ScanState>(ScanState.Scanning)
    val scanState: StateFlow<ScanState> = _scanState.asStateFlow()

    /** Remembered locally; neither captured photos nor OCR text are persisted. */
    val healthProfile: StateFlow<HealthProfile> = profileRepository.profile.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000),
        initialValue = HealthProfile(),
    )

    private val _fps = MutableStateFlow(0f)
    val fps: StateFlow<Float> = _fps.asStateFlow()

    // Match PreviewView's center-crop aspect ratio for overlay alignment.
    private val _liveFrameAspectRatio = MutableStateFlow(3f / 4f)
    val liveFrameAspectRatio: StateFlow<Float> = _liveFrameAspectRatio.asStateFlow()
    private var lastFrameTime = System.nanoTime()
    private var frameCount = 0
    private var stableLiveDetection: Detection? = null
    private var pendingLiveDetection: Detection? = null
    private var pendingLiveFrames = 0
    private var liveDetectionMisses = 0

    /** Updates the live framing overlay from the newest preview frame. */
    fun processFrame(imageProxy: ImageProxy) {
        if (_scanState.value !is ScanState.Scanning) {
            imageProxy.close()
            return
        }

        var bitmap: Bitmap? = null
        try {
            val rawBitmap = imageProxy.toBitmap()
            bitmap = rawBitmap.rotated(imageProxy.imageInfo.rotationDegrees)
            if (bitmap !== rawBitmap) rawBitmap.recycle()

            val bestDetection = detector.detect(bitmap).maxByOrNull { it.score }
            // Ignore a frame that finished after capture began.
            if (_scanState.value is ScanState.Scanning) {
                _liveFrameAspectRatio.value = bitmap.width.toFloat() / bitmap.height.coerceAtLeast(1)
                _detections.value = stabilizeLiveDetection(bestDetection)?.let(::listOf).orEmpty()
                updateFps()
            }
        } catch (_: Exception) {
            // Keep preview failures nonfatal.
        } finally {
            bitmap?.recycle()
            imageProxy.close()
        }
    }

    /** Runs detection on the captured image before crop review. */
    fun triggerScan(capturedBitmap: Bitmap) {
        if (_scanState.value !is ScanState.Scanning) return
        _scanState.value = ScanState.Capturing(capturedBitmap)
        clearLiveDetection()

        viewModelScope.launch {
            try {
                val bestDetection = withContext(Dispatchers.Default) {
                    detector.detect(capturedBitmap).maxByOrNull { it.score }
                }
                _scanState.value = ScanState.AdjustingCrop(
                    CropReview(
                        capturedBitmap = capturedBitmap,
                        initialCrop = bestDetection?.box?.expandedForOcr() ?: RectF(0f, 0f, 1f, 1f),
                        detectorFound = bestDetection != null,
                    ),
                )
            } catch (_: Exception) {
                _scanState.value = ScanState.Error(
                    message = "The label couldn't be read. Retake the photo with the label flat, well lit, and in focus.",
                )
            }
        }
    }

    /** Runs OCR only after the participant has corrected the suggested crop. */
    fun confirmCrop(cropReview: CropReview, selectedCrop: RectF) {
        if (_scanState.value !is ScanState.AdjustingCrop) return
        val normalizedCrop = selectedCrop.normalizedCrop()
        val cropped = cropBitmap(cropReview.capturedBitmap, normalizedCrop)
        val usedFullImageFallback = !cropReview.detectorFound || normalizedCrop.isNearlyFullImage()
        _scanState.value = ScanState.ExtractingText(cropped, usedFullImageFallback)

        viewModelScope.launch {
            try {
                val ocrResult = withContext(Dispatchers.Default) {
                    ocrProcessor.recognizeText(cropped)
                }
                _scanState.value = ScanState.Translating(cropped, ocrResult, usedFullImageFallback)

                val translationResult = withContext(Dispatchers.Default) {
                    translationManager.translateToEnglish(ocrResult.rawText)
                }
                _scanState.value = ScanState.ReviewNutrition(
                    NutritionReview(
                        croppedBitmap = cropped,
                        ocrResult = ocrResult,
                        translationResult = translationResult,
                        usedFullImageFallback = usedFullImageFallback,
                    ),
                )
            } catch (_: Exception) {
                _scanState.value = ScanState.Error(
                    message = "The selected area couldn't be read. Adjust the crop or retake the photo with the label flat, well lit, and in focus.",
                )
            }
        }
    }

    /** Sends reviewed label text and the selected profile to Gemini. */
    fun requestGuidance(review: NutritionReview, reviewedLabelText: String) {
        val state = _scanState.value as? ScanState.ReviewNutrition ?: return
        if (state.review != review) return

        val submittedText = reviewedLabelText.trim()
        if (submittedText.isBlank()) return
        val originalOcrText = review.ocrResult.rawText.trim()
            .takeIf { it.isNotEmpty() && it != submittedText }
        analyze(
            LabelAnalysisRequest(
                review = review,
                reviewedLabelText = submittedText,
                originalOcrText = originalOcrText,
                sourceLanguage = review.translationResult.detectedLanguage,
                profile = healthProfile.value,
            ),
        )
    }

    fun retryRecommendation() {
        val state = _scanState.value as? ScanState.Error ?: return
        state.retryRequest?.let(::analyze)
    }

    fun openProfileEditor() {
        if (_scanState.value is ScanState.Scanning) {
            _scanState.value = ScanState.ProfileEditor(healthProfile.value)
        }
    }

    fun saveProfile(profile: HealthProfile) {
        viewModelScope.launch {
            profileRepository.save(profile)
            _scanState.value = ScanState.Scanning
        }
    }

    fun clearProfile() {
        viewModelScope.launch {
            profileRepository.clear()
            _scanState.value = ScanState.ProfileEditor(HealthProfile())
        }
    }

    fun resetToScanning() {
        clearLiveDetection()
        _scanState.value = ScanState.Scanning
    }

    fun reportCaptureError() {
        _scanState.value = ScanState.Error(
            message = "The camera couldn't capture a photo. Please try again.",
        )
    }

    private fun analyze(request: LabelAnalysisRequest) {
        _scanState.value = ScanState.Analyzing(request)
        viewModelScope.launch {
            when (
                val result = geminiClient.analyzeLabel(
                    reviewedLabelText = request.reviewedLabelText,
                    originalOcrText = request.originalOcrText,
                    sourceLanguage = request.sourceLanguage,
                    profile = request.profile,
                    responseLanguage = "English",
                )
            ) {
                is GeminiResult.Success -> {
                    _scanState.value = ScanState.Results(request, result.advice)
                }
                is GeminiResult.Failure -> {
                    _scanState.value = ScanState.Error(
                        message = result.message,
                        retryable = result.retryable,
                        retryRequest = request,
                    )
                }
            }
        }
    }

    private fun cropBitmap(bitmap: Bitmap, crop: RectF): Bitmap {
        val left = (crop.left * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
        val top = (crop.top * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
        val right = (crop.right * bitmap.width).toInt().coerceIn(left + 1, bitmap.width)
        val bottom = (crop.bottom * bitmap.height).toInt().coerceIn(top + 1, bitmap.height)
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
    }

    /** Smooths the live preview overlay. */
    private fun stabilizeLiveDetection(candidate: Detection?): Detection? {
        if (candidate == null) {
            pendingLiveDetection = null
            pendingLiveFrames = 0
            if (stableLiveDetection != null && ++liveDetectionMisses >= 4) {
                stableLiveDetection = null
            }
            return stableLiveDetection
        }

        liveDetectionMisses = 0
        val stable = stableLiveDetection
        if (stable == null) {
            val pending = pendingLiveDetection
            if (pending != null && pending.box.iou(candidate.box) >= 0.35f) {
                pendingLiveFrames++
            } else {
                pendingLiveDetection = candidate
                pendingLiveFrames = 1
            }
            if (pendingLiveFrames >= 2) {
                stableLiveDetection = pendingLiveDetection
                pendingLiveDetection = null
                pendingLiveFrames = 0
            }
            return stableLiveDetection
        }

        if (stable.box.iou(candidate.box) >= 0.20f) {
            stableLiveDetection = Detection(
                box = RectF(
                    stable.box.left * 0.75f + candidate.box.left * 0.25f,
                    stable.box.top * 0.75f + candidate.box.top * 0.25f,
                    stable.box.right * 0.75f + candidate.box.right * 0.25f,
                    stable.box.bottom * 0.75f + candidate.box.bottom * 0.25f,
                ),
                score = stable.score * 0.75f + candidate.score * 0.25f,
            )
            pendingLiveDetection = null
            pendingLiveFrames = 0
        } else {
            val pending = pendingLiveDetection
            if (pending != null && pending.box.iou(candidate.box) >= 0.35f) {
                pendingLiveFrames++
            } else {
                pendingLiveDetection = candidate
                pendingLiveFrames = 1
            }
            if (pendingLiveFrames >= 3) {
                stableLiveDetection = pendingLiveDetection
                pendingLiveDetection = null
                pendingLiveFrames = 0
            }
        }
        return stableLiveDetection
    }

    private fun updateFps() {
        frameCount++
        val now = System.nanoTime()
        val elapsedSeconds = (now - lastFrameTime) / 1_000_000_000f
        if (elapsedSeconds >= 0.5f) {
            _fps.value = frameCount / elapsedSeconds.coerceAtLeast(0.001f)
            frameCount = 0
            lastFrameTime = now
        }
    }

    private fun clearLiveDetection() {
        _detections.value = emptyList()
        _fps.value = 0f
        _liveFrameAspectRatio.value = 3f / 4f
        stableLiveDetection = null
        pendingLiveDetection = null
        pendingLiveFrames = 0
        liveDetectionMisses = 0
        frameCount = 0
        lastFrameTime = System.nanoTime()
    }

    private fun Bitmap.rotated(rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return this
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    override fun onCleared() {
        detector.close()
        ocrProcessor.close()
        translationManager.close()
        geminiClient.close()
        super.onCleared()
    }
}

private fun RectF.expandedForOcr(): RectF {
    val marginX = width() * 0.06f
    val marginY = height() * 0.06f
    return RectF(left - marginX, top - marginY, right + marginX, bottom + marginY).normalizedCrop()
}

private fun RectF.normalizedCrop(): RectF {
    val minimumSize = 0.08f
    val left = left.coerceIn(0f, 1f - minimumSize)
    val top = top.coerceIn(0f, 1f - minimumSize)
    val right = right.coerceIn(left + minimumSize, 1f)
    val bottom = bottom.coerceIn(top + minimumSize, 1f)
    return RectF(left, top, right, bottom)
}

private fun RectF.isNearlyFullImage(): Boolean =
    left <= 0.01f && top <= 0.01f && right >= 0.99f && bottom >= 0.99f

private fun RectF.iou(other: RectF): Float {
    val left = maxOf(left, other.left)
    val top = maxOf(top, other.top)
    val right = minOf(right, other.right)
    val bottom = minOf(bottom, other.bottom)
    val intersection = maxOf(0f, right - left) * maxOf(0f, bottom - top)
    val union = width() * height() + other.width() * other.height() - intersection
    return intersection / maxOf(union, 1e-6f)
}
