package com.example.nutritionscanner.ocr

import android.graphics.Bitmap
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Text returned by the selected OCR recognizer. */
data class NutritionOcrResult(
    val rawText: String,
    val bestScript: String,    // e.g. "Latin", "Devanagari", "Chinese"
)

/** Runs bundled ML Kit script recognizers and selects the strongest result. */
class OcrProcessor {

    private data class RecognizerEntry(
        val name: String,
        val recognizer: TextRecognizer,
    )

    private val recognizers = listOf(
        RecognizerEntry("Latin", TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)),
        RecognizerEntry("Devanagari", TextRecognition.getClient(DevanagariTextRecognizerOptions.Builder().build())),
        RecognizerEntry("Chinese", TextRecognition.getClient(ChineseTextRecognizerOptions.Builder().build())),
        RecognizerEntry("Japanese", TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build())),
        RecognizerEntry("Korean", TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())),
    )

    /** Recognizes text in a cropped nutrition label. */
    suspend fun recognizeText(bitmap: Bitmap): NutritionOcrResult {
        val inputImage = InputImage.fromBitmap(bitmap, 0)

        var bestText = ""
        var bestScript = "Latin"

        for (entry in recognizers) {
            try {
                val text = runRecognizer(entry.recognizer, inputImage)
                if (text.length > bestText.length) {
                    bestText = text
                    bestScript = entry.name
                }
            } catch (_: Exception) {
            }
        }

        return NutritionOcrResult(
            rawText = bestText.ifEmpty { "No text detected" },
            bestScript = bestScript,
        )
    }

    private suspend fun runRecognizer(
        recognizer: TextRecognizer,
        image: InputImage,
    ): String = suspendCancellableCoroutine { cont ->
        recognizer.process(image)
            .addOnSuccessListener { visionText ->
                cont.resume(visionText.text)
            }
            .addOnFailureListener { e ->
                cont.resumeWithException(e)
            }
    }

    fun close() {
        recognizers.forEach { it.recognizer.close() }
    }
}
