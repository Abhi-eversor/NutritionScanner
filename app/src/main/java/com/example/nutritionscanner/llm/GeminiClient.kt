package com.example.nutritionscanner.llm

import com.example.nutritionscanner.BuildConfig
import com.example.nutritionscanner.profile.HealthProfile
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectBuilder
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

/** Sends reviewed nutrition facts and profile flags to Gemini. */
class GeminiClient {

    companion object {
        private const val MODEL = "gemini-3.5-flash"
        private const val ENDPOINT =
            "https://generativelanguage.googleapis.com/v1beta/models/$MODEL:generateContent"
        private const val MAX_OUTPUT_TOKENS = 512
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(45, TimeUnit.SECONDS)
        .writeTimeout(20, TimeUnit.SECONDS)
        .build()

    private val json = Json {
        ignoreUnknownKeys = true
        explicitNulls = false
    }

    /** Generates guidance from participant-confirmed nutrition facts. */
    suspend fun analyzeNutrition(
        facts: NutritionFacts,
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

        val request = Request.Builder()
            .url(ENDPOINT)
            // Keep the API key out of the URL.
            .header("x-goog-api-key", apiKey)
            .header("Accept", "application/json")
            .post(buildRequestBody(facts, profile, responseLanguage).toRequestBody(JSON_MEDIA_TYPE))
            .build()

        try {
            httpClient.newCall(request).execute().use { response ->
                val responseBody = response.body?.string().orEmpty()
                if (!response.isSuccessful) {
                    return@withContext httpFailure(response.code)
                }
                return@withContext parseResponse(responseBody)
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: IOException) {
            GeminiResult.Failure(
                reason = GeminiFailureReason.NETWORK,
                message = "Couldn't reach Gemini. Check the connection and try again.",
                retryable = true,
            )
        } catch (_: Exception) {
            GeminiResult.Failure(
                reason = GeminiFailureReason.UNEXPECTED,
                message = "Gemini couldn't complete the recommendation. Please try again.",
                retryable = true,
            )
        }
    }

    private fun buildRequestBody(
        facts: NutritionFacts,
        profile: HealthProfile,
        responseLanguage: String,
    ): String = buildJsonObject {
        putJsonArray("contents") {
            add(buildJsonObject {
                putJsonArray("parts") {
                    add(buildJsonObject {
                        put("text", buildPrompt(facts, profile, responseLanguage))
                    })
                }
            })
        }
        putJsonObject("generationConfig") {
            put("maxOutputTokens", MAX_OUTPUT_TOKENS)
            putJsonObject("responseFormat") {
                putJsonObject("text") {
                    put("mimeType", "application/json")
                    put("schema", recommendationSchema())
                }
            }
        }
    }.toString()

    private fun buildPrompt(
        facts: NutritionFacts,
        profile: HealthProfile,
        responseLanguage: String,
    ): String = buildString {
        appendLine("You provide general, non-diagnostic nutrition guidance for a research demo.")
        appendLine("Use only the confirmed nutrition facts and profile below. Never invent missing values, ingredients, or medical facts.")
        appendLine("Do not diagnose disease, prescribe treatment, give dosing advice, or claim that a food prevents or treats a condition.")
        appendLine("If a profile flag is relevant, phrase the caution as general information and recommend a qualified clinician for personal medical advice.")
        appendLine("Keep highlights and cautions concise. Use plain language.")
        appendLine("Write every user-visible string in ${responseLanguage.ifBlank { "English" }}.")
        appendLine()
        appendLine("CONFIRMED_NUTRITION_FACTS_JSON")
        appendLine(facts.toPromptJson())
        appendLine("END_CONFIRMED_NUTRITION_FACTS_JSON")
        appendLine()
        appendLine("HEALTH_PROFILE_JSON")
        appendLine(profile.toPromptJson())
        appendLine("END_HEALTH_PROFILE_JSON")
    }

    private fun NutritionFacts.toPromptJson(): JsonObject = buildJsonObject {
        putNullableString("servingSize", servingSize)
        putNullableNumber("caloriesKcal", caloriesKcal)
        putNullableNumber("totalFatGrams", totalFatGrams)
        putNullableNumber("saturatedFatGrams", saturatedFatGrams)
        putNullableNumber("transFatGrams", transFatGrams)
        putNullableNumber("cholesterolMg", cholesterolMg)
        putNullableNumber("sodiumMg", sodiumMg)
        putNullableNumber("totalCarbohydrateGrams", totalCarbohydrateGrams)
        putNullableNumber("dietaryFiberGrams", dietaryFiberGrams)
        putNullableNumber("totalSugarsGrams", totalSugarsGrams)
        putNullableNumber("addedSugarsGrams", addedSugarsGrams)
        putNullableNumber("proteinGrams", proteinGrams)
        putNullableNumber("potassiumMg", potassiumMg)
        putNullableNumber("calciumMg", calciumMg)
        putNullableNumber("ironMg", ironMg)
        putNullableString("ingredients", ingredients)
    }

    private fun HealthProfile.toPromptJson(): JsonObject = buildJsonObject {
        put("dietaryPreference", dietaryPreference.displayName)
        putJsonArray("allergies") {
            allergies.filter { it.isNotBlank() }.forEach { add(it.trim()) }
        }
        put("hasDiabetes", hasDiabetes)
        put("hasHypertension", hasHypertension)
        put("hasKidneyDisease", hasKidneyDisease)
        put("isPregnant", isPregnant)
        put("reducedSugar", reducedSugar)
        put("reducedSodium", reducedSodium)
    }

    private fun recommendationSchema(): JsonObject = buildJsonObject {
        put("type", "object")
        putJsonObject("properties") {
            putJsonObject("rating") {
                put("type", "string")
                putJsonArray("enum") {
                    add("HEALTHY")
                    add("MODERATE")
                    add("LIMIT")
                }
            }
            putStringArrayProperty("summary", "A concise label-grounded assessment.")
            putStringListProperty("highlights", "Two or three concise positive observations.")
            putStringListProperty("cautions", "Zero to three concise concerns or missing-data notes.")
            putStringArrayProperty("profileFit", "How the confirmed facts relate to the supplied profile.")
            putStringArrayProperty("recommendation", "One practical, non-diagnostic suggestion.")
            putStringArrayProperty("disclaimer", "A brief statement that this is general nutrition information, not medical advice.")
        }
        putJsonArray("required") {
            add("rating")
            add("summary")
            add("highlights")
            add("cautions")
            add("profileFit")
            add("recommendation")
            add("disclaimer")
        }
        put("additionalProperties", false)
    }

    private fun JsonObjectBuilder.putStringArrayProperty(name: String, description: String) {
        putJsonObject(name) {
            put("type", "string")
            put("description", description)
        }
    }

    private fun JsonObjectBuilder.putStringListProperty(name: String, description: String) {
        putJsonObject(name) {
            put("type", "array")
            put("description", description)
            putJsonObject("items") { put("type", "string") }
            put("maxItems", 3)
        }
    }

    private fun JsonObjectBuilder.putNullableString(name: String, value: String?) {
        val normalized = value?.trim()?.takeIf { it.isNotEmpty() }
        if (normalized == null) {
            put(name, JsonNull)
        } else {
            put(name, normalized)
        }
    }

    private fun JsonObjectBuilder.putNullableNumber(name: String, value: Double?) {
        if (value == null) {
            put(name, JsonNull)
        } else {
            put(name, value)
        }
    }

    private fun parseResponse(body: String): GeminiResult {
        return try {
            val envelope = json.parseToJsonElement(body).jsonObject
            val promptBlocked = envelope["promptFeedback"]
                ?.jsonObject
                ?.get("blockReason")
                ?.jsonPrimitive
                ?.contentOrNull
                ?.isNotBlank() == true
            if (promptBlocked) {
                return GeminiResult.Failure(
                    reason = GeminiFailureReason.BLOCKED,
                    message = "Gemini couldn't generate guidance for this request.",
                    retryable = false,
                )
            }

            val candidate = envelope["candidates"]
                ?.jsonArray
                ?.firstOrNull()
                ?.jsonObject
                ?: return GeminiResult.Failure(
                    reason = GeminiFailureReason.EMPTY_RESPONSE,
                    message = "Gemini returned no recommendation. Please try again.",
                    retryable = true,
                )

            val finishReason = candidate["finishReason"]?.jsonPrimitive?.contentOrNull
            if (finishReason == "SAFETY" || finishReason == "BLOCKLIST") {
                return GeminiResult.Failure(
                    reason = GeminiFailureReason.BLOCKED,
                    message = "Gemini couldn't generate guidance for this request.",
                    retryable = false,
                )
            }

            val structuredText = candidate["content"]
                ?.jsonObject
                ?.get("parts")
                ?.jsonArray
                ?.mapNotNull { part -> part.jsonObject["text"]?.jsonPrimitive?.contentOrNull }
                ?.joinToString(separator = "")
                ?.trim()
                .orEmpty()

            if (structuredText.isBlank()) {
                return GeminiResult.Failure(
                    reason = GeminiFailureReason.EMPTY_RESPONSE,
                    message = "Gemini returned an empty recommendation. Please try again.",
                    retryable = true,
                )
            }

            val recommendation = json.decodeFromString<GeminiRecommendation>(structuredText)
            validateRecommendation(recommendation)
        } catch (_: Exception) {
            GeminiResult.Failure(
                reason = GeminiFailureReason.INVALID_RESPONSE,
                message = "Gemini returned a response the app couldn't read. Please try again.",
                retryable = true,
            )
        }
    }

    private fun validateRecommendation(recommendation: GeminiRecommendation): GeminiResult {
        val hasRequiredText = listOf(
            recommendation.summary,
            recommendation.profileFit,
            recommendation.recommendation,
            recommendation.disclaimer,
        ).all { it.isNotBlank() }
        if (!hasRequiredText) {
            return GeminiResult.Failure(
                reason = GeminiFailureReason.INVALID_RESPONSE,
                message = "Gemini returned an incomplete recommendation. Please try again.",
                retryable = true,
            )
        }
        return GeminiResult.Success(recommendation)
    }

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
            message = "Gemini couldn't process this request. Please review the label and try again.",
            retryable = true,
        )
    }

    fun close() {
        httpClient.dispatcher.executorService.shutdown()
        httpClient.connectionPool.evictAll()
    }
}
