package com.example.nutritionscanner.profile

/** User-editable flags used to tailor label guidance. */
data class HealthProfile(
    val dietaryPreference: DietaryPreference = DietaryPreference.NONE,
    val allergies: List<String> = emptyList(),
    val hasDiabetes: Boolean = false,
    val hasHypertension: Boolean = false,
    val hasKidneyDisease: Boolean = false,
    val isPregnant: Boolean = false,
    val reducedSugar: Boolean = false,
    val reducedSodium: Boolean = false,
) {
    fun toPromptContext(): String = buildList {
        if (dietaryPreference != DietaryPreference.NONE) {
            add("Dietary preference: ${dietaryPreference.displayName}")
        }
        allergies.filter { it.isNotBlank() }.takeIf { it.isNotEmpty() }?.let {
            add("Reported allergies: ${it.joinToString()}")
        }
        if (hasDiabetes) add("Health flag: diabetes")
        if (hasHypertension) add("Health flag: hypertension")
        if (hasKidneyDisease) add("Health flag: kidney disease")
        if (isPregnant) add("Health flag: pregnancy")
        if (reducedSugar) add("Preference: reduced sugar")
        if (reducedSodium) add("Preference: reduced sodium")
    }.ifEmpty { listOf("No health-profile flags were supplied") }.joinToString("; ")
}

enum class DietaryPreference(val displayName: String) {
    NONE("No dietary preference"),
    VEGETARIAN("Vegetarian"),
    VEGAN("Vegan"),
    HALAL("Halal"),
    KOSHER("Kosher"),
}
