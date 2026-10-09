package com.example.nutritionscanner.translation

import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Result of language detection and optional translation. */
data class TranslationResult(
    val originalText: String,
    val translatedText: String,
    val detectedLanguage: String,
    val wasTranslated: Boolean,
)

/** Uses ML Kit to detect the OCR language and translate supported text to English. */
class TranslationManager {

    private val languageIdentifier = LanguageIdentification.getClient()
    private val translatorCache = mutableMapOf<String, Translator>()

    /** Translates supported non-English text to English. */
    suspend fun translateToEnglish(text: String): TranslationResult {
        if (text.isBlank()) {
            return TranslationResult(text, text, "en", false)
        }

        val detectedLang = identifyLanguage(text)

        if (detectedLang == "en" || detectedLang == "und") {
            return TranslationResult(
                originalText = text,
                translatedText = text,
                detectedLanguage = detectedLang,
                wasTranslated = false,
            )
        }

        val sourceTag = TranslateLanguage.fromLanguageTag(detectedLang)
        if (sourceTag == null) {
            // Unsupported language — return original text
            return TranslationResult(
                originalText = text,
                translatedText = text,
                detectedLanguage = detectedLang,
                wasTranslated = false,
            )
        }

        val translator = getOrCreateTranslator(sourceTag)

        downloadModelIfNeeded(translator)

        val translated = translate(translator, text)

        return TranslationResult(
            originalText = text,
            translatedText = translated,
            detectedLanguage = detectedLang,
            wasTranslated = true,
        )
    }

    private suspend fun identifyLanguage(text: String): String =
        suspendCancellableCoroutine { cont ->
            languageIdentifier.identifyLanguage(text)
                .addOnSuccessListener { languageCode ->
                    cont.resume(languageCode)
                }
                .addOnFailureListener {
                    cont.resume("und")
                }
        }

    private fun getOrCreateTranslator(sourceLanguage: String): Translator {
        return translatorCache.getOrPut(sourceLanguage) {
            val options = TranslatorOptions.Builder()
                .setSourceLanguage(sourceLanguage)
                .setTargetLanguage(TranslateLanguage.ENGLISH)
                .build()
            Translation.getClient(options)
        }
    }

    private suspend fun downloadModelIfNeeded(translator: Translator): Unit =
        suspendCancellableCoroutine { cont ->
            val conditions = DownloadConditions.Builder()
                .build()
            translator.downloadModelIfNeeded(conditions)
                .addOnSuccessListener { cont.resume(Unit) }
                .addOnFailureListener { e -> cont.resumeWithException(e) }
        }

    private suspend fun translate(translator: Translator, text: String): String =
        suspendCancellableCoroutine { cont ->
            translator.translate(text)
                .addOnSuccessListener { translated -> cont.resume(translated) }
                .addOnFailureListener { e -> cont.resumeWithException(e) }
        }

    fun close() {
        languageIdentifier.close()
        translatorCache.values.forEach { it.close() }
        translatorCache.clear()
    }
}
