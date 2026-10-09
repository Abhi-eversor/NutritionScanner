package com.example.nutritionscanner.llm

/** Parses English or translated label text before the participant confirms it. */
object NutritionFactsParser {

    fun parse(labelText: String): NutritionFacts {
        val normalized = labelText
            .replace('\u00a0', ' ')
            .replace(',', '.')
            .replace(Regex("[ \\t]+"), " ")

        return NutritionFacts(
            servingSize = servingSize(labelText),
            caloriesKcal = amountWithoutUnit(normalized, "calories?"),
            totalFatGrams = grams(normalized, "(?:total )?fat"),
            saturatedFatGrams = grams(normalized, "saturated fat"),
            transFatGrams = grams(normalized, "trans fat"),
            cholesterolMg = milligrams(normalized, "cholesterol"),
            sodiumMg = milligrams(normalized, "sodium"),
            totalCarbohydrateGrams = grams(normalized, "(?:total )?(?:carbohydrate|carbohydrates|carbs)"),
            dietaryFiberGrams = grams(normalized, "(?:dietary )?fibre|(?:dietary )?fiber"),
            totalSugarsGrams = grams(normalized, "(?:total )?sugars?"),
            addedSugarsGrams = grams(normalized, "added sugars?"),
            proteinGrams = grams(normalized, "protein"),
            potassiumMg = milligrams(normalized, "potassium"),
            calciumMg = milligrams(normalized, "calcium"),
            ironMg = milligrams(normalized, "iron"),
            ingredients = ingredients(labelText),
        )
    }

    private fun servingSize(text: String): String? {
        val match = Regex(
            pattern = "(?im)^\\s*serving\\s+size\\s*[:\\-]?\\s*(.+?)\\s*$",
        ).find(text) ?: return null
        return match.groupValues[1].trim().takeIf { it.isNotEmpty() }
    }

    private fun ingredients(text: String): String? {
        val match = Regex(
            pattern = "(?is)\\bingredients?\\s*[:\\-]\\s*(.+?)(?=\\b(?:contains|allergen|nutrition facts)\\b|$)",
        ).find(text) ?: return null
        return match.groupValues[1].trim().takeIf { it.isNotEmpty() }
    }

    private fun amountWithoutUnit(text: String, field: String): Double? {
        val match = Regex("(?i)\\b$field\\b[^0-9]{0,16}(\\d+(?:\\.\\d+)?)").find(text)
            ?: return null
        return match.groupValues[1].toDoubleOrNull()
    }

    private fun grams(text: String, field: String): Double? = amountWithUnit(text, field)?.let { (value, unit) ->
        when (unit.lowercase()) {
            "g" -> value
            "mg" -> value / 1_000.0
            "mcg", "ug", "\u00b5g" -> value / 1_000_000.0
            else -> null
        }
    }

    private fun milligrams(text: String, field: String): Double? = amountWithUnit(text, field)?.let { (value, unit) ->
        when (unit.lowercase()) {
            "mg" -> value
            "g" -> value * 1_000.0
            "mcg", "ug", "\u00b5g" -> value / 1_000.0
            else -> null
        }
    }

    private fun amountWithUnit(text: String, field: String): Pair<Double, String>? {
        val match = Regex(
            pattern = "(?i)\\b(?:$field)\\b[^0-9]{0,16}(\\d+(?:\\.\\d+)?)\\s*(mg|mcg|\u00b5g|ug|g)\\b",
        ).find(text) ?: return null
        val value = match.groupValues[1].toDoubleOrNull() ?: return null
        return value to match.groupValues[2]
    }
}
