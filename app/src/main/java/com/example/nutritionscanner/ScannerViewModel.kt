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
import com.example.nutritionscanner.llm.GeminiRecommendation
import com.example.nutritionscanner.llm.GeminiResult
import com.example.nutritionscanner.llm.NutritionFacts
import com.example.nutritionscanner.llm.NutritionFactsParser
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
import kotlin.math.max

/** A completed OCR/translation pass waiting for the participant to confirm values. */
data class NutritionReview(
    val croppedBitmap: Bitmap,
    val ocrResult: NutritionOcrResult,
    val translationResult: TranslationResult,
    val facts: NutritionFacts,
    val usedFullImageFallback: Boolean,
)

/** UI states for capture, confirmation, profile editing, and recommendations. */
sealed class ScanState {
    data object Scanning : ScanState()
    data class Capturing(val capturedBitmap: Bitmap) : ScanState()
    data class ExtractingText(val croppedBitmap: Bitmap, val usedFullImageFallback: Boolean) : ScanState()
    data class Translating(
        val croppedBitmap: Bitmap,
        val ocrResult: NutritionOcrResult,
        val usedFullImageFallback: Boolean,
    ) : ScanState()
    data class ReviewNutrition(val review: NutritionReview) : ScanState()
    data class Analyzing(val review: NutritionReview) : ScanState()
    data class Results(
        val review: NutritionReview,
        val recommendation: GeminiRecommendation,
    ) : ScanState()
    data class ProfileEditor(val profile: HealthProfile) : ScanState()
    data class Error(
        val message: String,
        val retryable: Boolean = false,
        val review: NutritionReview? = null,
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
    private var lastFrameTime = System.nanoTime()
    private var frameCount = 0

    /** Runs low-latency live detection for the preview overlay. */
    fun processFrame(imageProxy: ImageProxy) {
        if (_scanState.value !is ScanState.Scanning) {
            imageProxy.close()
            return
        }

        try {
            val bitmap = imageProxy.toBitmap().rotated(imageProxy.imageInfo.rotationDegrees)
            val results = detector.detect(bitmap)
            _detections.value = results
            updateFps()
        } catch (_: Exception) {
            // Keep the preview stream alive.
        } finally {
            imageProxy.close()
        }
    }

    /** Runs detection on the captured image before OCR. */
    fun triggerScan(capturedBitmap: Bitmap) {
        if (_scanState.value !is ScanState.Scanning) return
        _scanState.value = ScanState.Capturing(capturedBitmap)
        _detections.value = emptyList()

        viewModelScope.launch {
            try {
                val bestDetection = withContext(Dispatchers.Default) {
                    detector.detect(capturedBitmap).maxByOrNull { it.score }
                }
                val usedFallback = bestDetection == null
                val cropped = if (bestDetection == null) {
                    capturedBitmap
                } else {
                    cropDetection(capturedBitmap, bestDetection.box)
                }
                _scanState.value = ScanState.ExtractingText(cropped, usedFallback)

                val ocrResult = withContext(Dispatchers.Default) {
                    ocrProcessor.recognizeText(cropped)
                }
                _scanState.value = ScanState.Translating(cropped, ocrResult, usedFallback)

                val translationResult = withContext(Dispatchers.Default) {
                    translationManager.translateToEnglish(ocrResult.rawText)
                }
                val labelText = if (translationResult.wasTranslated) {
                    translationResult.translatedText
                } else {
                    translationResult.originalText
                }
                val review = NutritionReview(
                    croppedBitmap = cropped,
                    ocrResult = ocrResult,
                    translationResult = translationResult,
                    facts = NutritionFactsParser.parse(labelText),
                    usedFullImageFallback = usedFallback,
                )
                _scanState.value = ScanState.ReviewNutrition(review)
            } catch (_: Exception) {
                _scanState.value = ScanState.Error(
                    message = "The label couldn't be read. Retake the photo with the label flat, well lit, and in focus.",
                )
            }
        }
    }

    /** Sends only participant-confirmed values to Gemini. */
    fun confirmNutritionFacts(facts: NutritionFacts) {
        val state = _scanState.value as? ScanState.ReviewNutrition ?: return
        analyze(state.review.copy(facts = facts))
    }

    fun retryRecommendation() {
        val state = _scanState.value as? ScanState.Error ?: return
        state.review?.let(::analyze)
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
        _detections.value = emptyList()
        _scanState.value = ScanState.Scanning
    }

    fun reportCaptureError() {
        _scanState.value = ScanState.Error(
            message = "The camera couldn't capture a photo. Please try again.",
        )
    }

    private fun analyze(review: NutritionReview) {
        _scanState.value = ScanState.Analyzing(review)
        viewModelScope.launch {
            when (
                val result = geminiClient.analyzeNutrition(
                    facts = review.facts,
                    profile = healthProfile.value,
                    responseLanguage = "English",
                )
            ) {
                is GeminiResult.Success -> {
                    _scanState.value = ScanState.Results(review, result.recommendation)
                }
                is GeminiResult.Failure -> {
                    _scanState.value = ScanState.Error(
                        message = result.message,
                        retryable = result.retryable,
                        review = review,
                    )
                }
            }
        }
    }

    private fun cropDetection(bitmap: Bitmap, box: RectF): Bitmap {
        val marginX = box.width() * 0.06f
        val marginY = box.height() * 0.06f
        val left = ((box.left - marginX) * bitmap.width).toInt().coerceIn(0, bitmap.width - 1)
        val top = ((box.top - marginY) * bitmap.height).toInt().coerceIn(0, bitmap.height - 1)
        val right = ((box.right + marginX) * bitmap.width).toInt().coerceIn(left + 1, bitmap.width)
        val bottom = ((box.bottom + marginY) * bitmap.height).toInt().coerceIn(top + 1, bitmap.height)
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
    }

    private fun updateFps() {
        frameCount++
        val now = System.nanoTime()
        val elapsedSeconds = (now - lastFrameTime) / 1_000_000_000f
        if (elapsedSeconds >= 0.5f) {
            _fps.value = frameCount / max(elapsedSeconds, 0.001f)
            frameCount = 0
            lastFrameTime = now
        }
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
