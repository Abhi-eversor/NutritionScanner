package com.example.nutritionscanner.llm

/** A complete, plain-text Gemini response shown without semantic parsing. */
data class GeminiAdvice(
    val text: String,
)

sealed interface GeminiResult {
    data class Success(val advice: GeminiAdvice) : GeminiResult

    data class Failure(
        val reason: GeminiFailureReason,
        val message: String,
        val retryable: Boolean,
    ) : GeminiResult
}

enum class GeminiFailureReason {
    MISSING_API_KEY,
    AUTHENTICATION,
    NETWORK,
    RATE_LIMITED,
    SERVICE_UNAVAILABLE,
    REQUEST_REJECTED,
    BLOCKED,
    EMPTY_RESPONSE,
    INVALID_RESPONSE,
    UNEXPECTED,
}
