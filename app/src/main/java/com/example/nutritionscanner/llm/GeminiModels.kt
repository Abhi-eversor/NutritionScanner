package com.example.nutritionscanner.llm

import kotlinx.serialization.Serializable

/**
 * Nutrition values confirmed by the participant after OCR review.
 * All nutrient fields are nullable because labels differ by market and format.
 */
data class NutritionFacts(
    val servingSize: String? = null,
    val caloriesKcal: Double? = null,
    val totalFatGrams: Double? = null,
    val saturatedFatGrams: Double? = null,
    val transFatGrams: Double? = null,
    val cholesterolMg: Double? = null,
    val sodiumMg: Double? = null,
    val totalCarbohydrateGrams: Double? = null,
    val dietaryFiberGrams: Double? = null,
    val totalSugarsGrams: Double? = null,
    val addedSugarsGrams: Double? = null,
    val proteinGrams: Double? = null,
    val potassiumMg: Double? = null,
    val calciumMg: Double? = null,
    val ironMg: Double? = null,
    val ingredients: String? = null,
)

@Serializable
enum class RecommendationRating {
    HEALTHY,
    MODERATE,
    LIMIT,
}

/** A schema-backed response suitable for predictable Compose rendering. */
@Serializable
data class GeminiRecommendation(
    val rating: RecommendationRating,
    val summary: String,
    val highlights: List<String>,
    val cautions: List<String>,
    val profileFit: String,
    val recommendation: String,
    val disclaimer: String,
)

sealed interface GeminiResult {
    data class Success(val recommendation: GeminiRecommendation) : GeminiResult

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
