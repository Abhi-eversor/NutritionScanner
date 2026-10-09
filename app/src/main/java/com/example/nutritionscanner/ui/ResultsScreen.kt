package com.example.nutritionscanner.ui

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Translate
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.PopupProperties
import com.example.nutritionscanner.NutritionReview
import com.example.nutritionscanner.ScanState
import com.example.nutritionscanner.llm.GeminiRecommendation
import com.example.nutritionscanner.llm.NutritionFacts
import com.example.nutritionscanner.llm.RecommendationRating
import com.example.nutritionscanner.profile.DietaryPreference
import com.example.nutritionscanner.profile.HealthProfile
import com.example.nutritionscanner.theme.DarkCard
import com.example.nutritionscanner.theme.DarkSurface
import com.example.nutritionscanner.theme.DarkSurfaceVariant
import com.example.nutritionscanner.theme.Emerald400
import com.example.nutritionscanner.theme.Emerald500
import com.example.nutritionscanner.theme.Emerald700
import com.example.nutritionscanner.theme.ModerateYellow
import com.example.nutritionscanner.theme.TextMuted
import com.example.nutritionscanner.theme.TextPrimary
import com.example.nutritionscanner.theme.TextSecondary
import com.example.nutritionscanner.theme.UnhealthyRed

/** Full-screen workflow sheet shown over the camera preview. */
@Composable
fun ResultsScreen(
    state: ScanState,
    profile: HealthProfile,
    onBack: () -> Unit,
    onConfirmNutritionFacts: (NutritionFacts) -> Unit,
    onRetryRecommendation: () -> Unit,
    onSaveProfile: (HealthProfile) -> Unit,
    onClearProfile: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(DarkSurface)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
    ) {
        Header(title = state.title(), onBack = onBack)
        Spacer(Modifier.height(16.dp))

        when (state) {
            is ScanState.Capturing -> ProgressCard(
                "Finding the nutrition panel…",
                "Checking the high-resolution photo before OCR.",
            )
            is ScanState.ExtractingText -> {
                CroppedImageCard(state.croppedBitmap, state.usedFullImageFallback)
                Spacer(Modifier.height(16.dp))
                ProgressCard("Reading the label…", "Running on-device OCR across bundled scripts.")
            }
            is ScanState.Translating -> {
                CroppedImageCard(state.croppedBitmap, state.usedFullImageFallback)
                Spacer(Modifier.height(16.dp))
                OcrResultCard(state.ocrResult.rawText, state.ocrResult.bestScript)
                Spacer(Modifier.height(16.dp))
                ProgressCard("Preparing nutrition fields…", "Detecting the label language and translating locally if needed.")
            }
            is ScanState.ReviewNutrition -> NutritionReviewContent(
                review = state.review,
                profile = profile,
                onConfirm = onConfirmNutritionFacts,
            )
            is ScanState.Analyzing -> {
                CroppedImageCard(state.review.croppedBitmap, state.review.usedFullImageFallback)
                Spacer(Modifier.height(16.dp))
                ProfileCard(profile)
                Spacer(Modifier.height(16.dp))
                ProgressCard(
                    "Creating a label-grounded recommendation…",
                    "Gemini receives only the values you confirmed and the saved profile flags.",
                )
            }
            is ScanState.Results -> ResultsContent(state.review, state.recommendation, profile)
            is ScanState.ProfileEditor -> HealthProfileEditor(
                initialProfile = state.profile,
                onSave = onSaveProfile,
                onClear = onClearProfile,
            )
            is ScanState.Error -> ErrorContent(
                message = state.message,
                canRetry = state.retryable && state.review != null,
                onRetry = onRetryRecommendation,
            )
            ScanState.Scanning -> Unit
        }
        Spacer(Modifier.height(28.dp))
    }
}

@Composable
private fun Header(title: String, onBack: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        IconButton(onClick = onBack) {
            Icon(Icons.Default.ArrowBack, contentDescription = "Back to camera", tint = TextPrimary)
        }
        Text(
            title,
            style = MaterialTheme.typography.headlineSmall,
            color = TextPrimary,
            modifier = Modifier.padding(start = 8.dp),
        )
    }
}

@Composable
private fun NutritionReviewContent(
    review: NutritionReview,
    profile: HealthProfile,
    onConfirm: (NutritionFacts) -> Unit,
) {
    var servingSize by remember(review) { mutableStateOf(review.facts.servingSize.orEmpty()) }
    var calories by remember(review) { mutableStateOf(review.facts.caloriesKcal.editorValue()) }
    var totalFat by remember(review) { mutableStateOf(review.facts.totalFatGrams.editorValue()) }
    var saturatedFat by remember(review) { mutableStateOf(review.facts.saturatedFatGrams.editorValue()) }
    var cholesterol by remember(review) { mutableStateOf(review.facts.cholesterolMg.editorValue()) }
    var sodium by remember(review) { mutableStateOf(review.facts.sodiumMg.editorValue()) }
    var carbohydrate by remember(review) { mutableStateOf(review.facts.totalCarbohydrateGrams.editorValue()) }
    var fibre by remember(review) { mutableStateOf(review.facts.dietaryFiberGrams.editorValue()) }
    var sugars by remember(review) { mutableStateOf(review.facts.totalSugarsGrams.editorValue()) }
    var addedSugars by remember(review) { mutableStateOf(review.facts.addedSugarsGrams.editorValue()) }
    var protein by remember(review) { mutableStateOf(review.facts.proteinGrams.editorValue()) }
    var potassium by remember(review) { mutableStateOf(review.facts.potassiumMg.editorValue()) }
    var ingredients by remember(review) { mutableStateOf(review.facts.ingredients.orEmpty()) }

    CroppedImageCard(review.croppedBitmap, review.usedFullImageFallback)
    Spacer(Modifier.height(16.dp))
    OcrResultCard(review.ocrResult.rawText, review.ocrResult.bestScript)
    if (review.translationResult.wasTranslated) {
        Spacer(Modifier.height(16.dp))
        TranslationCard(review.translationResult.detectedLanguage, review.translationResult.translatedText)
    }
    Spacer(Modifier.height(16.dp))
    ProfileCard(profile)
    Spacer(Modifier.height(16.dp))

    Card(colors = CardDefaults.cardColors(containerColor = DarkCard), modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Confirm nutrition facts", style = MaterialTheme.typography.titleLarge, color = Emerald400)
            Spacer(Modifier.height(6.dp))
            Text(
                "Correct anything OCR misread. Only these confirmed values are sent for the recommendation.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
            Spacer(Modifier.height(14.dp))
            OutlinedTextField(
                value = servingSize,
                onValueChange = { servingSize = it },
                label = { Text("Serving size") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            NutritionNumberField("Calories (kcal)", calories) { calories = it }
            NutritionNumberField("Total fat (g)", totalFat) { totalFat = it }
            NutritionNumberField("Saturated fat (g)", saturatedFat) { saturatedFat = it }
            NutritionNumberField("Cholesterol (mg)", cholesterol) { cholesterol = it }
            NutritionNumberField("Sodium (mg)", sodium) { sodium = it }
            NutritionNumberField("Carbohydrate (g)", carbohydrate) { carbohydrate = it }
            NutritionNumberField("Dietary fibre (g)", fibre) { fibre = it }
            NutritionNumberField("Total sugars (g)", sugars) { sugars = it }
            NutritionNumberField("Added sugars (g)", addedSugars) { addedSugars = it }
            NutritionNumberField("Protein (g)", protein) { protein = it }
            NutritionNumberField("Potassium (mg)", potassium) { potassium = it }
            OutlinedTextField(
                value = ingredients,
                onValueChange = { ingredients = it },
                label = { Text("Ingredients (optional)") },
                minLines = 2,
                modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
            )
            Spacer(Modifier.height(16.dp))
            Button(
                onClick = {
                    onConfirm(
                        review.facts.copy(
                            servingSize = servingSize.trim().ifBlank { null },
                            caloriesKcal = calories.toDoubleOrNull(),
                            totalFatGrams = totalFat.toDoubleOrNull(),
                            saturatedFatGrams = saturatedFat.toDoubleOrNull(),
                            cholesterolMg = cholesterol.toDoubleOrNull(),
                            sodiumMg = sodium.toDoubleOrNull(),
                            totalCarbohydrateGrams = carbohydrate.toDoubleOrNull(),
                            dietaryFiberGrams = fibre.toDoubleOrNull(),
                            totalSugarsGrams = sugars.toDoubleOrNull(),
                            addedSugarsGrams = addedSugars.toDoubleOrNull(),
                            proteinGrams = protein.toDoubleOrNull(),
                            potassiumMg = potassium.toDoubleOrNull(),
                            ingredients = ingredients.trim().ifBlank { null },
                            // Fields not exposed in this confirmation form remain
                            // unknown rather than being sent as unreviewed OCR.
                            transFatGrams = null,
                            calciumMg = null,
                            ironMg = null,
                        ),
                    )
                },
                colors = ButtonDefaults.buttonColors(containerColor = Emerald500),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Get general recommendation")
            }
        }
    }
}

@Composable
private fun NutritionNumberField(label: String, value: String, onValueChange: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = { input -> onValueChange(input.filter { it.isDigit() || it == '.' }) },
        label = { Text(label) },
        singleLine = true,
        keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = KeyboardType.Decimal),
        modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
    )
}

@Composable
private fun HealthProfileEditor(
    initialProfile: HealthProfile,
    onSave: (HealthProfile) -> Unit,
    onClear: () -> Unit,
) {
    var dietaryPreference by remember(initialProfile) { mutableStateOf(initialProfile.dietaryPreference) }
    var allergies by remember(initialProfile) { mutableStateOf(initialProfile.allergies.joinToString(", ")) }
    var diabetes by remember(initialProfile) { mutableStateOf(initialProfile.hasDiabetes) }
    var hypertension by remember(initialProfile) { mutableStateOf(initialProfile.hasHypertension) }
    var kidneyDisease by remember(initialProfile) { mutableStateOf(initialProfile.hasKidneyDisease) }
    var pregnancy by remember(initialProfile) { mutableStateOf(initialProfile.isPregnant) }
    var reducedSugar by remember(initialProfile) { mutableStateOf(initialProfile.reducedSugar) }
    var reducedSodium by remember(initialProfile) { mutableStateOf(initialProfile.reducedSodium) }
    var dietMenuExpanded by remember { mutableStateOf(false) }

    Card(colors = CardDefaults.cardColors(containerColor = DarkCard), modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Saved health profile", style = MaterialTheme.typography.titleLarge, color = Emerald400)
            Spacer(Modifier.height(6.dp))
            Text(
                "This is stored only on this device and can be edited or cleared before each scan.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
            )
            Spacer(Modifier.height(16.dp))
            Text("Dietary preference", color = TextPrimary, style = MaterialTheme.typography.labelLarge)
            Spacer(Modifier.height(6.dp))
            OutlinedButton(onClick = { dietMenuExpanded = true }, modifier = Modifier.fillMaxWidth()) {
                Text(dietaryPreference.displayName)
            }
            DropdownMenu(
                expanded = dietMenuExpanded,
                onDismissRequest = { dietMenuExpanded = false },
                properties = PopupProperties(focusable = true),
            ) {
                DietaryPreference.entries.forEach { preference ->
                    DropdownMenuItem(
                        text = { Text(preference.displayName) },
                        onClick = {
                            dietaryPreference = preference
                            dietMenuExpanded = false
                        },
                    )
                }
            }
            OutlinedTextField(
                value = allergies,
                onValueChange = { allergies = it },
                label = { Text("Allergies (comma separated, optional)") },
                modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            )
            Spacer(Modifier.height(12.dp))
            ProfileSwitch("Diabetes", diabetes) { diabetes = it }
            ProfileSwitch("Hypertension", hypertension) { hypertension = it }
            ProfileSwitch("Kidney disease", kidneyDisease) { kidneyDisease = it }
            ProfileSwitch("Pregnancy", pregnancy) { pregnancy = it }
            ProfileSwitch("Prefer reduced sugar", reducedSugar) { reducedSugar = it }
            ProfileSwitch("Prefer reduced sodium", reducedSodium) { reducedSodium = it }
            Spacer(Modifier.height(14.dp))
            Button(
                onClick = {
                    onSave(
                        HealthProfile(
                            dietaryPreference = dietaryPreference,
                            allergies = allergies.split(',').map(String::trim).filter(String::isNotBlank),
                            hasDiabetes = diabetes,
                            hasHypertension = hypertension,
                            hasKidneyDisease = kidneyDisease,
                            isPregnant = pregnancy,
                            reducedSugar = reducedSugar,
                            reducedSodium = reducedSodium,
                        ),
                    )
                },
                colors = ButtonDefaults.buttonColors(containerColor = Emerald500),
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Save profile") }
            TextButton(onClick = onClear, modifier = Modifier.align(Alignment.CenterHorizontally)) {
                Text("Clear saved profile", color = ModerateYellow)
            }
        }
    }
}

@Composable
private fun ProfileSwitch(label: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
        modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp),
    ) {
        Text(label, color = TextPrimary, style = MaterialTheme.typography.bodyMedium)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun ResultsContent(
    review: NutritionReview,
    recommendation: GeminiRecommendation,
    profile: HealthProfile,
) {
    CroppedImageCard(review.croppedBitmap, review.usedFullImageFallback)
    Spacer(Modifier.height(16.dp))
    ProfileCard(profile)
    Spacer(Modifier.height(16.dp))
    RecommendationCard(recommendation)
    Spacer(Modifier.height(16.dp))
    OcrResultCard(review.ocrResult.rawText, review.ocrResult.bestScript)
    if (review.translationResult.wasTranslated) {
        Spacer(Modifier.height(16.dp))
        TranslationCard(review.translationResult.detectedLanguage, review.translationResult.translatedText)
    }
}

@Composable
private fun CroppedImageCard(bitmap: Bitmap, usedFullImageFallback: Boolean) {
    Card(colors = CardDefaults.cardColors(containerColor = DarkCard), modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Captured label", style = MaterialTheme.typography.titleMedium, color = Emerald400)
            if (usedFullImageFallback) {
                Spacer(Modifier.height(6.dp))
                Text(
                    "The detector did not confirm a panel, so OCR used the full photo. Verify the fields carefully.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ModerateYellow,
                )
            }
            Spacer(Modifier.height(12.dp))
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Nutrition label photo",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxWidth().heightIn(max = 240.dp).clip(MaterialTheme.shapes.medium),
            )
        }
    }
}

@Composable
private fun OcrResultCard(text: String, script: String) {
    Card(colors = CardDefaults.cardColors(containerColor = DarkCard), modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                Text("Extracted text", style = MaterialTheme.typography.titleMedium, color = Emerald400)
                Spacer(Modifier.weight(1f))
                Surface(color = Emerald700, shape = MaterialTheme.shapes.small) {
                    Text(script, style = MaterialTheme.typography.labelSmall, color = TextPrimary, modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp))
                }
            }
            Spacer(Modifier.height(10.dp))
            Surface(color = DarkSurfaceVariant, shape = MaterialTheme.shapes.small, modifier = Modifier.fillMaxWidth()) {
                Text(text, style = MaterialTheme.typography.bodyMedium, color = TextSecondary, modifier = Modifier.padding(12.dp))
            }
        }
    }
}

@Composable
private fun TranslationCard(sourceLanguage: String, text: String) {
    Card(colors = CardDefaults.cardColors(containerColor = DarkCard), modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Translate, contentDescription = null, tint = ModerateYellow, modifier = Modifier.size(20.dp))
                Spacer(Modifier.size(8.dp))
                Text("Translated from $sourceLanguage", style = MaterialTheme.typography.titleMedium, color = ModerateYellow)
            }
            Spacer(Modifier.height(10.dp))
            Text(text, style = MaterialTheme.typography.bodyMedium, color = TextPrimary)
        }
    }
}

@Composable
private fun ProfileCard(profile: HealthProfile) {
    Card(colors = CardDefaults.cardColors(containerColor = DarkCard), modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Profile used for this recommendation", style = MaterialTheme.typography.titleMedium, color = Emerald400)
            Spacer(Modifier.height(6.dp))
            Text(profile.toPromptContext(), style = MaterialTheme.typography.bodySmall, color = TextSecondary)
        }
    }
}

@Composable
private fun RecommendationCard(recommendation: GeminiRecommendation) {
    val ratingColor = when (recommendation.rating) {
        RecommendationRating.HEALTHY -> Emerald400
        RecommendationRating.MODERATE -> ModerateYellow
        RecommendationRating.LIMIT -> UnhealthyRed
    }
    Card(
        colors = CardDefaults.cardColors(containerColor = DarkCard),
        border = BorderStroke(1.dp, ratingColor),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("General nutrition guidance", style = MaterialTheme.typography.titleLarge, color = ratingColor)
            Text(recommendation.rating.name.replace('_', ' '), style = MaterialTheme.typography.labelLarge, color = ratingColor)
            Spacer(Modifier.height(10.dp))
            Text(recommendation.summary, style = MaterialTheme.typography.bodyLarge, color = TextPrimary)
            RecommendationSection("Highlights", recommendation.highlights, Emerald400)
            RecommendationSection("Considerations", recommendation.cautions, ModerateYellow)
            Spacer(Modifier.height(12.dp))
            Text("Profile context", style = MaterialTheme.typography.titleSmall, color = TextPrimary)
            Text(recommendation.profileFit, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
            Spacer(Modifier.height(12.dp))
            Text("Practical next step", style = MaterialTheme.typography.titleSmall, color = TextPrimary)
            Text(recommendation.recommendation, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
            Spacer(Modifier.height(14.dp))
            Text(recommendation.disclaimer, style = MaterialTheme.typography.bodySmall, color = TextMuted)
        }
    }
}

@Composable
private fun RecommendationSection(title: String, items: List<String>, color: androidx.compose.ui.graphics.Color) {
    if (items.isEmpty()) return
    Spacer(Modifier.height(12.dp))
    Text(title, style = MaterialTheme.typography.titleSmall, color = color)
    items.forEach { item ->
        Text("• $item", style = MaterialTheme.typography.bodyMedium, color = TextSecondary, modifier = Modifier.padding(top = 3.dp))
    }
}

@Composable
private fun ProgressCard(title: String, subtitle: String) {
    Card(colors = CardDefaults.cardColors(containerColor = DarkCard), modifier = Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(20.dp)) {
            CircularProgressIndicator(color = Emerald500, strokeWidth = 3.dp, modifier = Modifier.size(32.dp))
            Spacer(Modifier.size(16.dp))
            Column {
                Text(title, style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                Spacer(Modifier.height(3.dp))
                Text(subtitle, style = MaterialTheme.typography.bodySmall, color = TextMuted)
            }
        }
    }
}

@Composable
private fun ErrorContent(message: String, canRetry: Boolean, onRetry: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(containerColor = DarkCard),
        border = BorderStroke(1.dp, UnhealthyRed),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Scan needs attention", style = MaterialTheme.typography.titleMedium, color = UnhealthyRed)
            Spacer(Modifier.height(8.dp))
            Text(message, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
            if (canRetry) {
                Spacer(Modifier.height(14.dp))
                Button(onClick = onRetry, colors = ButtonDefaults.buttonColors(containerColor = Emerald500)) {
                    Text("Retry recommendation")
                }
            }
        }
    }
}

private fun ScanState.title(): String = when (this) {
    is ScanState.ProfileEditor -> "Health profile"
    is ScanState.ReviewNutrition -> "Review label"
    is ScanState.Results -> "Recommendation"
    is ScanState.Error -> "Scan issue"
    else -> "Nutrition scan"
}

private fun Double?.editorValue(): String = when {
    this == null -> ""
    this % 1.0 == 0.0 -> toInt().toString()
    else -> toString()
}
