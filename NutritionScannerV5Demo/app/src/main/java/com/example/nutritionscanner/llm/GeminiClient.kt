package com.example.nutritionscanner.llm

import com.example.nutritionscanner.BuildConfig
import com.example.nutritionscanner.profile.HealthProfile
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

/** Sends reviewed label text and profile context to Gemini. */
class GeminiClient {

    companion object {
        private const val MODEL = "gemini-3.1-flash-lite"
        private const val ENDPOINT =
            "https://generativelanguage.googleapis.com/v1beta/models/$MODEL:generateContent"
        private const val MAX_OUTPUT_TOKENS = 700
        private const val MAX_REQUEST_ATTEMPTS = 2
        private const val RETRY_DELAY_MS = 750L
        private const val MAX_LABEL_TEXT_CHARS = 8_000
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(75, TimeUnit.SECONDS)
        .writeTimeout(20, TimeUnit.SECONDS)
        .callTimeout(90, TimeUnit.SECONDS)
        .build()

    suspend fun analyzeLabel(
        reviewedLabelText: String,
        originalOcrText: String?,
        sourceLanguage: String,
        profile: HealthProfile,
        responseLanguage: String,
    ): GeminiResult = withContext(Dispatchers.IO) {
        val apiKey = BuildConfig.GEMINI_API_KEY.trim()
        if (apiKey.isEmpty()) {
            return@withContext GeminiResult.Failure(
                reason = GeminiFailureReason.MISSING_API_KEY,
                message = "Gemini is not configured for this demo build.",
                retryable = false,
            )
        }

        when (
            val response = execute(
                apiKey,
                buildRequestBody(
                    reviewedLabelText = reviewedLabelText,
                    originalOcrText = originalOcrText,
                    sourceLanguage = sourceLanguage,
                    profile = profile,
                    responseLanguage = responseLanguage,
                ),
            )
        ) {
            is ApiCallResult.Failure -> response.failure
            is ApiCallResult.Success -> when (val parsed = GeminiResponseParser.parseApiResponse(response.body)) {
                is GeminiResponseParser.EnvelopeResult.Success -> GeminiResult.Success(parsed.advice)
                is GeminiResponseParser.EnvelopeResult.Failure -> parsed.toGeminiFailure()
            }
        }
    }

    /** Internal for request-body tests; it never includes API credentials. */
    internal fun buildRequestBody(
        reviewedLabelText: String,
        originalOcrText: String?,
        sourceLanguage: String,
        profile: HealthProfile,
        responseLanguage: String,
    ): String = buildJsonObject {
        putJsonArray("contents") {
            add(buildJsonObject {
                putJsonArray("parts") {
                    add(buildJsonObject {
                        put(
                            "text",
                            buildPrompt(
                                reviewedLabelText = reviewedLabelText,
                                originalOcrText = originalOcrText,
                                sourceLanguage = sourceLanguage,
                                profile = profile,
                                responseLanguage = responseLanguage,
                            ),
                        )
                    })
                }
            })
        }
        putJsonObject("generationConfig") {
            put("maxOutputTokens", MAX_OUTPUT_TOKENS)
            put("temperature", 0.35)
        }
    }.toString()

    private fun buildPrompt(
        reviewedLabelText: String,
        originalOcrText: String?,
        sourceLanguage: String,
        profile: HealthProfile,
        responseLanguage: String,
    ): String = buildString {
        appendLine("You provide personalized, label-grounded nutrition guidance for a research demo.")
        appendLine("The supplied label text comes from OCR and may have rows in an unusual order, missing punctuation, or recognition errors.")
        appendLine("Use it cautiously. Do not invent nutrients, ingredients, serving sizes, medical facts, or certainty that are not supported by the text.")
        appendLine("Use the health-profile flags directly when they are relevant: connect confirmed values to practical choices such as portion size, frequency, or a lower-sodium/lower-sugar alternative.")
        appendLine("Give a clear, label-specific recommendation rather than generic wellness advice. State what looks favourable, what deserves attention, and one practical next step.")
        appendLine("When a relevant value is missing, say that it cannot be assessed from this label instead of guessing.")
        appendLine("Do not diagnose disease, prescribe treatment, give dosing advice, or claim that a food prevents or treats a condition.")
        appendLine("Write a practical response of about 150 to 220 words in ${responseLanguage.ifBlank { "English" }}.")
        appendLine("Use short plain-text paragraphs and optional bullet lines. Do not return JSON, Markdown headings, code fences, or a rating label.")
        appendLine("End with this exact final line: This is general nutrition information, not medical advice; check with a qualified doctor or dietitian for advice tailored to you.")
        appendLine()
        appendLine("USER_REVIEWED_LABEL_TEXT")
        appendLine(reviewedLabelText.trim().take(MAX_LABEL_TEXT_CHARS))
        appendLine("END_USER_REVIEWED_LABEL_TEXT")
        appendLine()
        appendLine("OCR_SOURCE_LANGUAGE")
        appendLine(sourceLanguage.ifBlank { "unknown" })
        appendLine("END_OCR_SOURCE_LANGUAGE")

        originalOcrText
            ?.trim()
            ?.takeIf { it.isNotEmpty() && !it.equals(reviewedLabelText.trim(), ignoreCase = false) }
            ?.let { original ->
                appendLine()
                appendLine("ORIGINAL_OCR_TEXT")
                appendLine(original.take(MAX_LABEL_TEXT_CHARS))
                appendLine("END_ORIGINAL_OCR_TEXT")
            }

        appendLine()
        appendLine("HEALTH_PROFILE_JSON")
        appendLine(profile.toPromptJson())
        appendLine("END_HEALTH_PROFILE_JSON")
    }

    /** Runs the primary request with a small retry policy for transient failures. */
    private fun execute(apiKey: String, requestBody: String): ApiCallResult {
        repeat(MAX_REQUEST_ATTEMPTS) { attempt ->
            when (val result = executeOnce(apiKey, requestBody)) {
                is ApiCallResult.Success -> return result
                is ApiCallResult.Failure -> {
                    if (!result.failure.retryable || attempt == MAX_REQUEST_ATTEMPTS - 1) return result
                    Thread.sleep(RETRY_DELAY_MS)
                }
            }
        }
        return ApiCallResult.Failure(unexpectedFailure())
    }

    private fun executeOnce(apiKey: String, requestBody: String): ApiCallResult = try {
        val request = Request.Builder()
            .url(ENDPOINT)
            .header("x-goog-api-key", apiKey)
            .header("Accept", "application/json")
            .post(requestBody.toRequestBody(JSON_MEDIA_TYPE))
            .build()
        httpClient.newCall(request).execute().use { response ->
            if (response.isSuccessful) {
                ApiCallResult.Success(response.body?.string().orEmpty())
            } else {
                // Do not expose backend error bodies because they may echo prompt context.
                ApiCallResult.Failure(httpFailure(response.code))
            }
        }
    } catch (cancelled: CancellationException) {
        throw cancelled
    } catch (_: IOException) {
        ApiCallResult.Failure(
            GeminiResult.Failure(
                reason = GeminiFailureReason.NETWORK,
                message = "Couldn't reach Gemini. Check the connection and try again.",
                retryable = true,
            ),
        )
    } catch (_: Exception) {
        ApiCallResult.Failure(unexpectedFailure())
    }

    private fun HealthProfile.toPromptJson(): JsonObject = buildJsonObject {
        put("dietaryPreference", dietaryPreference.displayName)
        putJsonArray("allergies") {
            allergies.filter { it.isNotBlank() }.forEach { add(JsonPrimitive(it.trim())) }
        }
        put("hasDiabetes", hasDiabetes)
        put("hasHypertension", hasHypertension)
        put("hasKidneyDisease", hasKidneyDisease)
        put("isPregnant", isPregnant)
        put("reducedSugar", reducedSugar)
        put("reducedSodium", reducedSodium)
    }

    private fun GeminiResponseParser.EnvelopeResult.Failure.toGeminiFailure() = GeminiResult.Failure(
        reason = reason,
        message = message,
        retryable = retryable,
    )

    private fun httpFailure(code: Int): GeminiResult.Failure = when (code) {
        401, 403 -> GeminiResult.Failure(
            reason = GeminiFailureReason.AUTHENTICATION,
            message = "Gemini authorization failed for this demo build.",
            retryable = false,
        )
        429 -> GeminiResult.Failure(
            reason = GeminiFailureReason.RATE_LIMITED,
            message = "Gemini is busy right now. Please try again shortly.",
            retryable = true,
        )
        in 500..599 -> GeminiResult.Failure(
            reason = GeminiFailureReason.SERVICE_UNAVAILABLE,
            message = "Gemini is temporarily unavailable. Please try again.",
            retryable = true,
        )
        else -> GeminiResult.Failure(
            reason = GeminiFailureReason.REQUEST_REJECTED,
            message = "Gemini couldn't process this label. Please review the text and try again.",
            retryable = true,
        )
    }

    private fun unexpectedFailure() = GeminiResult.Failure(
        reason = GeminiFailureReason.UNEXPECTED,
        message = "Gemini couldn't complete the guidance. Please try again.",
        retryable = true,
    )

    private sealed interface ApiCallResult {
        data class Success(val body: String) : ApiCallResult
        data class Failure(val failure: GeminiResult.Failure) : ApiCallResult
    }

    fun close() {
        httpClient.dispatcher.executorService.shutdown()
        httpClient.connectionPool.evictAll()
    }

}
