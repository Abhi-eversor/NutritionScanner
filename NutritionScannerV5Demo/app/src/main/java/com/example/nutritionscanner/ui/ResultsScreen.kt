package com.example.nutritionscanner.ui

import android.graphics.Bitmap
import android.graphics.RectF
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
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
import androidx.compose.material3.OutlinedTextFieldDefaults
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.PopupProperties
import com.example.nutritionscanner.CropReview
import com.example.nutritionscanner.LabelAnalysisRequest
import com.example.nutritionscanner.NutritionReview
import com.example.nutritionscanner.ScanState
import com.example.nutritionscanner.llm.GeminiAdvice
import com.example.nutritionscanner.profile.DietaryPreference
import com.example.nutritionscanner.profile.HealthProfile
import com.example.nutritionscanner.theme.Apricot100
import com.example.nutritionscanner.theme.Apricot700
import com.example.nutritionscanner.theme.Basil100
import com.example.nutritionscanner.theme.Basil600
import com.example.nutritionscanner.theme.CanvasIvory
import com.example.nutritionscanner.theme.Ink
import com.example.nutritionscanner.theme.InkMuted
import com.example.nutritionscanner.theme.LineSoft
import com.example.nutritionscanner.theme.NutritionSystemBars
import com.example.nutritionscanner.theme.SurfaceWhite

/** White editorial workflow sheet shown over the camera preview. */
@Composable
fun ResultsScreen(
    state: ScanState,
    profile: HealthProfile,
    onBack: () -> Unit,
    onConfirmCrop: (CropReview, RectF) -> Unit,
    onRequestGuidance: (NutritionReview, String) -> Unit,
    onRetryRecommendation: () -> Unit,
    onSaveProfile: (HealthProfile) -> Unit,
    onClearProfile: () -> Unit,
    modifier: Modifier = Modifier,
) {
    NutritionSystemBars(color = CanvasIvory, useDarkIcons = true)
    Column(
        modifier = modifier
            .fillMaxSize()
            .background(CanvasIvory)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 14.dp),
    ) {
        Header(title = state.title(), onBack = onBack)
        Spacer(Modifier.height(22.dp))

        when (state) {
            is ScanState.Capturing -> ProgressCard(
                title = "Finding the label",
                detail = "Checking the captured photo before reading it.",
            )
            is ScanState.AdjustingCrop -> CropReviewContent(state.cropReview, onConfirmCrop)
            is ScanState.ExtractingText -> {
                LabelImageCard(state.croppedBitmap, state.usedFullImageFallback)
                Spacer(Modifier.height(16.dp))
                ProgressCard("Reading the label", "Recognising text on this device.")
            }
            is ScanState.Translating -> {
                LabelImageCard(state.croppedBitmap, state.usedFullImageFallback)
                Spacer(Modifier.height(16.dp))
                ProgressCard("Preparing the text", "Checking whether a local English translation is useful.")
            }
            is ScanState.ReviewNutrition -> ReviewRecognizedTextContent(
                review = state.review,
                profile = profile,
                onRequestGuidance = onRequestGuidance,
            )
            is ScanState.Analyzing -> AnalyzingContent(state.request)
            is ScanState.Results -> ResultsContent(state.request, state.advice)
            is ScanState.ProfileEditor -> HealthProfileEditor(
                initialProfile = state.profile,
                onSave = onSaveProfile,
                onClear = onClearProfile,
            )
            is ScanState.Error -> ErrorContent(
                message = state.message,
                canRetry = state.retryable && state.retryRequest != null,
                onRetry = onRetryRecommendation,
            )
            ScanState.Scanning -> Unit
        }
        Spacer(Modifier.height(32.dp))
    }
}

@Composable
private fun Header(title: String, onBack: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        IconButton(onClick = onBack) {
            Icon(
                Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "Back to camera",
                tint = Ink,
            )
        }
        Column(modifier = Modifier.padding(start = 8.dp)) {
            Text("NUTRITION SCAN", style = MaterialTheme.typography.labelMedium, color = Basil600)
            Text(title, style = MaterialTheme.typography.headlineMedium, color = Ink)
        }
    }
}

@Composable
private fun CropReviewContent(
    cropReview: CropReview,
    onConfirm: (CropReview, RectF) -> Unit,
) {
    var selectedCrop by remember(cropReview) { mutableStateOf(RectF(cropReview.initialCrop)) }

    EditorialCard {
        Text("Fine-tune the label", style = MaterialTheme.typography.headlineSmall, color = Ink)
        Spacer(Modifier.height(6.dp))
        Text(
            text = if (cropReview.detectorFound) {
                "We found a likely nutrition panel. Adjust the frame if it needs a little help."
            } else {
                "Select the nutrition panel before continuing."
            },
            style = MaterialTheme.typography.bodyMedium,
            color = InkMuted,
        )
        Spacer(Modifier.height(18.dp))
        AdjustableCropPreview(
            bitmap = cropReview.capturedBitmap,
            initialCrop = cropReview.initialCrop,
            onCropChanged = { selectedCrop = RectF(it) },
            modifier = Modifier
                .fillMaxWidth()
                .height(360.dp)
                .clip(MaterialTheme.shapes.large),
        )
        Spacer(Modifier.height(18.dp))
        PrimaryButton("Use this area") { onConfirm(cropReview, selectedCrop) }
    }
}

@Composable
private fun ReviewRecognizedTextContent(
    review: NutritionReview,
    profile: HealthProfile,
    onRequestGuidance: (NutritionReview, String) -> Unit,
) {
    val preferredText = remember(review) {
        if (review.translationResult.wasTranslated) {
            review.translationResult.translatedText
        } else {
            review.ocrResult.rawText
        }
    }
    var editableText by remember(review) { mutableStateOf(preferredText) }
    var originalExpanded by remember(review) { mutableStateOf(false) }

    LabelImageCard(review.croppedBitmap, review.usedFullImageFallback)
    Spacer(Modifier.height(16.dp))
    EditorialCard {
        Text("What we read", style = MaterialTheme.typography.headlineSmall, color = Ink)
        Spacer(Modifier.height(6.dp))
        Text(
            "This text is sent to Gemini, so correct only anything that is clearly misread.",
            style = MaterialTheme.typography.bodyMedium,
            color = InkMuted,
        )
        Spacer(Modifier.height(14.dp))
        OutlinedTextField(
            value = editableText,
            onValueChange = { editableText = it },
            modifier = Modifier.fillMaxWidth(),
            minLines = 7,
            maxLines = 11,
            label = { Text("Recognized label text") },
            colors = editorialTextFieldColors(),
        )
        if (review.translationResult.wasTranslated) {
            Spacer(Modifier.height(8.dp))
            TextButton(onClick = { originalExpanded = !originalExpanded }) {
                Text(if (originalExpanded) "Hide original OCR" else "View original recognized text")
                Spacer(Modifier.width(4.dp))
                Icon(
                    if (originalExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp),
                )
            }
            if (originalExpanded) {
                SourceTextBlock(
                    title = "Original OCR · ${review.ocrResult.bestScript}",
                    text = review.ocrResult.rawText,
                )
            }
        }
        Spacer(Modifier.height(16.dp))
        ProfileSummary(profile)
        Spacer(Modifier.height(16.dp))
        Text(
            "The reviewed text and your selected profile are sent to Gemini. Your photo stays on this device.",
            style = MaterialTheme.typography.bodySmall,
            color = InkMuted,
        )
        Spacer(Modifier.height(14.dp))
        PrimaryButton(
            label = "Get guidance",
            enabled = editableText.isNotBlank(),
        ) {
            onRequestGuidance(review, editableText)
        }
    }
}

@Composable
private fun AnalyzingContent(request: LabelAnalysisRequest) {
    LabelImageCard(request.review.croppedBitmap, request.review.usedFullImageFallback)
    Spacer(Modifier.height(16.dp))
    EditorialCard(tint = Basil100) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(color = Basil600, strokeWidth = 3.dp, modifier = Modifier.size(30.dp))
            Spacer(Modifier.width(16.dp))
            Column {
                Text("Looking at your label", style = MaterialTheme.typography.titleLarge, color = Ink)
                Spacer(Modifier.height(4.dp))
                Text(
                    "Gemini is using the text you reviewed and your saved profile.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = InkMuted,
                )
            }
        }
    }
}

@Composable
private fun ResultsContent(request: LabelAnalysisRequest, advice: GeminiAdvice) {
    LabelImageCard(request.review.croppedBitmap, request.review.usedFullImageFallback)
    Spacer(Modifier.height(16.dp))
    EditorialCard(tint = Basil100) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Surface(color = SurfaceWhite, shape = MaterialTheme.shapes.medium) {
                Icon(
                    Icons.Default.AutoAwesome,
                    contentDescription = null,
                    tint = Basil600,
                    modifier = Modifier.padding(10.dp).size(22.dp),
                )
            }
            Spacer(Modifier.width(12.dp))
            Column {
                Text("YOUR LABEL, INTERPRETED", style = MaterialTheme.typography.labelMedium, color = Basil600)
                Text("A practical read", style = MaterialTheme.typography.headlineSmall, color = Ink)
            }
        }
        Spacer(Modifier.height(18.dp))
        SelectionContainer {
            Text(advice.text, style = MaterialTheme.typography.bodyLarge, color = Ink)
        }
        Spacer(Modifier.height(18.dp))
        Surface(color = SurfaceWhite, shape = MaterialTheme.shapes.medium, modifier = Modifier.fillMaxWidth()) {
            Text(
                "General nutrition information only — not medical advice.",
                style = MaterialTheme.typography.bodySmall,
                color = InkMuted,
                modifier = Modifier.padding(12.dp),
            )
        }
    }
    Spacer(Modifier.height(16.dp))
    ProfileSummary(request.profile)
    Spacer(Modifier.height(16.dp))
    SourceTextDisclosure(request)
}

@Composable
private fun SourceTextDisclosure(request: LabelAnalysisRequest) {
    var expanded by remember(request) { mutableStateOf(false) }
    EditorialCard {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Source text", style = MaterialTheme.typography.titleMedium, color = Ink)
                Text("The text used to generate this guidance", style = MaterialTheme.typography.bodySmall, color = InkMuted)
            }
            TextButton(onClick = { expanded = !expanded }) {
                Text(if (expanded) "Hide" else "View")
                Icon(
                    if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                    contentDescription = null,
                )
            }
        }
        if (expanded) {
            Spacer(Modifier.height(12.dp))
            SourceTextBlock("Reviewed label text", request.reviewedLabelText)
            request.originalOcrText?.let { original ->
                Spacer(Modifier.height(12.dp))
                SourceTextBlock("Original OCR", original)
            }
        }
    }
}

@Composable
private fun SourceTextBlock(title: String, text: String) {
    Surface(color = CanvasIvory, shape = MaterialTheme.shapes.medium, modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(title, style = MaterialTheme.typography.labelMedium, color = Basil600)
            Spacer(Modifier.height(6.dp))
            SelectionContainer {
                Text(text, style = MaterialTheme.typography.bodyMedium, color = InkMuted)
            }
        }
    }
}

@Composable
private fun LabelImageCard(bitmap: Bitmap, usedFullImageFallback: Boolean) {
    EditorialCard {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Text("Scanned label", style = MaterialTheme.typography.titleMedium, color = Ink)
            Spacer(Modifier.weight(1f))
            if (!usedFullImageFallback) {
                Surface(color = Basil100, shape = MaterialTheme.shapes.small) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 9.dp, vertical = 5.dp)) {
                        Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Basil600, modifier = Modifier.size(15.dp))
                        Spacer(Modifier.width(4.dp))
                        Text("Label found", style = MaterialTheme.typography.labelMedium, color = Basil600)
                    }
                }
            }
        }
        if (usedFullImageFallback) {
            Spacer(Modifier.height(6.dp))
            Text("The full photo was used. Review the recognized text carefully.", style = MaterialTheme.typography.bodySmall, color = Apricot700)
        }
        Spacer(Modifier.height(12.dp))
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = "Scanned nutrition label",
            contentScale = ContentScale.Fit,
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(max = 280.dp)
                .clip(MaterialTheme.shapes.large),
        )
    }
}

@Composable
private fun ProfileSummary(profile: HealthProfile) {
    Surface(color = CanvasIvory, shape = MaterialTheme.shapes.medium, modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text("Profile used", style = MaterialTheme.typography.labelMedium, color = Basil600)
            Spacer(Modifier.height(4.dp))
            Text(profile.toPromptContext(), style = MaterialTheme.typography.bodyMedium, color = InkMuted)
        }
    }
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
    var menuExpanded by remember { mutableStateOf(false) }

    EditorialCard {
        Text("Your food profile", style = MaterialTheme.typography.headlineSmall, color = Ink)
        Spacer(Modifier.height(6.dp))
        Text("Saved only on this phone. Use it to tailor general label guidance.", style = MaterialTheme.typography.bodyMedium, color = InkMuted)
        Spacer(Modifier.height(18.dp))
        Text("Dietary preference", style = MaterialTheme.typography.labelLarge, color = Ink)
        Spacer(Modifier.height(6.dp))
        OutlinedButton(onClick = { menuExpanded = true }, modifier = Modifier.fillMaxWidth()) {
            Text(dietaryPreference.displayName)
        }
        DropdownMenu(
            expanded = menuExpanded,
            onDismissRequest = { menuExpanded = false },
            properties = PopupProperties(focusable = true),
        ) {
            DietaryPreference.entries.forEach { preference ->
                DropdownMenuItem(
                    text = { Text(preference.displayName) },
                    onClick = {
                        dietaryPreference = preference
                        menuExpanded = false
                    },
                )
            }
        }
        Spacer(Modifier.height(14.dp))
        OutlinedTextField(
            value = allergies,
            onValueChange = { allergies = it },
            label = { Text("Allergies or ingredients to avoid") },
            minLines = 2,
            modifier = Modifier.fillMaxWidth(),
            colors = editorialTextFieldColors(),
        )
        Spacer(Modifier.height(14.dp))
        Text("Health considerations", style = MaterialTheme.typography.labelLarge, color = Ink)
        ProfileSwitch("Diabetes", diabetes) { diabetes = it }
        ProfileSwitch("Hypertension", hypertension) { hypertension = it }
        ProfileSwitch("Kidney disease", kidneyDisease) { kidneyDisease = it }
        ProfileSwitch("Pregnancy", pregnancy) { pregnancy = it }
        Spacer(Modifier.height(10.dp))
        Text("Preferences", style = MaterialTheme.typography.labelLarge, color = Ink)
        ProfileSwitch("Prefer lower sugar", reducedSugar) { reducedSugar = it }
        ProfileSwitch("Prefer lower sodium", reducedSodium) { reducedSodium = it }
        Spacer(Modifier.height(18.dp))
        PrimaryButton("Save profile") {
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
        }
        Spacer(Modifier.height(8.dp))
        TextButton(onClick = onClear, modifier = Modifier.fillMaxWidth()) {
            Text("Clear saved profile", color = Apricot700)
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
        Text(label, color = Ink, style = MaterialTheme.typography.bodyMedium)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}

@Composable
private fun ProgressCard(title: String, detail: String) {
    EditorialCard(tint = Basil100) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(color = Basil600, strokeWidth = 3.dp, modifier = Modifier.size(30.dp))
            Spacer(Modifier.width(16.dp))
            Column {
                Text(title, style = MaterialTheme.typography.titleLarge, color = Ink)
                Spacer(Modifier.height(3.dp))
                Text(detail, style = MaterialTheme.typography.bodyMedium, color = InkMuted)
            }
        }
    }
}

@Composable
private fun ErrorContent(message: String, canRetry: Boolean, onRetry: () -> Unit) {
    EditorialCard(tint = Apricot100) {
        Text("This scan needs a second look", style = MaterialTheme.typography.headlineSmall, color = Ink)
        Spacer(Modifier.height(8.dp))
        Text(message, style = MaterialTheme.typography.bodyMedium, color = InkMuted)
        if (canRetry) {
            Spacer(Modifier.height(16.dp))
            PrimaryButton("Try Gemini again", onClick = onRetry)
        }
    }
}

@Composable
private fun EditorialCard(
    modifier: Modifier = Modifier,
    tint: Color = SurfaceWhite,
    content: @Composable ColumnScope.() -> Unit,
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = tint),
        shape = MaterialTheme.shapes.large,
        border = BorderStroke(1.dp, LineSoft),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(20.dp), content = content)
    }
}

@Composable
private fun PrimaryButton(label: String, enabled: Boolean = true, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        enabled = enabled,
        colors = ButtonDefaults.buttonColors(containerColor = Basil600, contentColor = Color.White),
        shape = MaterialTheme.shapes.medium,
        modifier = Modifier.fillMaxWidth().height(52.dp),
    ) {
        Text(label, style = MaterialTheme.typography.labelLarge)
    }
}

@Composable
private fun editorialTextFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor = Basil600,
    focusedLabelColor = Basil600,
    unfocusedBorderColor = LineSoft,
    focusedTextColor = Ink,
    unfocusedTextColor = Ink,
    cursorColor = Basil600,
    focusedContainerColor = SurfaceWhite,
    unfocusedContainerColor = SurfaceWhite,
)

private fun ScanState.title(): String = when (this) {
    is ScanState.ProfileEditor -> "Food profile"
    is ScanState.AdjustingCrop -> "Adjust label"
    is ScanState.ReviewNutrition -> "Review recognized text"
    is ScanState.Results -> "Your guidance"
    is ScanState.Error -> "Try again"
    else -> "Nutrition scan"
}
