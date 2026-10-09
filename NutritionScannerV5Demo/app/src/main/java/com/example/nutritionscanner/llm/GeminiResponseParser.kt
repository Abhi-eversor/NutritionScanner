package com.example.nutritionscanner.llm

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import java.util.Locale

/** Extracts displayable text from a Gemini GenerateContent response. */
internal object GeminiResponseParser {

    private val json = Json { ignoreUnknownKeys = true }

    sealed interface EnvelopeResult {
        data class Success(val advice: GeminiAdvice) : EnvelopeResult
        data class Failure(
            val reason: GeminiFailureReason,
            val message: String,
            val retryable: Boolean,
        ) : EnvelopeResult
    }

    fun parseApiResponse(body: String): EnvelopeResult = try {
        val envelope = json.parseToJsonElement(body).jsonObject
        val promptBlocked = envelope["promptFeedback"]
            ?.asObjectOrNull()
            ?.get("blockReason")
            ?.asText()
            ?.isNotBlank() == true
        if (promptBlocked) return blockedFailure()

        val candidates = envelope["candidates"]?.asArrayOrNull()
            ?: return emptyFailure()

        var sawBlockedCandidate = false
        candidates.forEach { candidateElement ->
            val candidate = candidateElement.asObjectOrNull() ?: return@forEach
            val finishReason = candidate["finishReason"]?.asText()?.uppercase(Locale.US)
            if (finishReason in setOf("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT")) {
                sawBlockedCandidate = true
                return@forEach
            }

            val text = candidate["content"]
                ?.asObjectOrNull()
                ?.get("parts")
                ?.asArrayOrNull()
                ?.mapNotNull { part -> part.asObjectOrNull()?.get("text")?.asText() }
                ?.joinToString(separator = "\n")
                ?.let(::cleanDisplayText)
                .orEmpty()
            if (text.isNotBlank()) return EnvelopeResult.Success(GeminiAdvice(text))
        }

        if (sawBlockedCandidate) blockedFailure() else emptyFailure()
    } catch (_: Exception) {
        EnvelopeResult.Failure(
            reason = GeminiFailureReason.INVALID_RESPONSE,
            message = "Gemini returned a response the app could not read. Please try again.",
            retryable = true,
        )
    }

    /** Removes transport-level whitespace and enclosing code fences. */
    private fun cleanDisplayText(text: String): String = text
        .replace("\r\n", "\n")
        .trim()
        .removeSurroundingCodeFence()
        .trim()

    private fun String.removeSurroundingCodeFence(): String {
        if (!startsWith("```") || !endsWith("```")) return this
        val lines = lines()
        return lines.drop(1).dropLast(1).joinToString("\n")
    }

    private fun blockedFailure() = EnvelopeResult.Failure(
        reason = GeminiFailureReason.BLOCKED,
        message = "Gemini could not generate guidance for this label.",
        retryable = false,
    )

    private fun emptyFailure() = EnvelopeResult.Failure(
        reason = GeminiFailureReason.EMPTY_RESPONSE,
        message = "Gemini returned no guidance. Please try again.",
        retryable = true,
    )

    private fun kotlinx.serialization.json.JsonElement?.asObjectOrNull(): JsonObject? = this as? JsonObject

    private fun kotlinx.serialization.json.JsonElement?.asArrayOrNull(): JsonArray? = this as? JsonArray

    private fun kotlinx.serialization.json.JsonElement?.asText(): String? = when (this) {
        is JsonPrimitive -> contentOrNull
        else -> null
    }
}
