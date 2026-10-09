package com.example.nutritionscanner.theme

import android.app.Activity
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.material3.Shapes
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.core.view.WindowCompat

private val NutritionLightColors = lightColorScheme(
    primary = Basil600,
    onPrimary = Color.White,
    primaryContainer = Basil100,
    onPrimaryContainer = Basil700,
    secondary = Apricot700,
    onSecondary = Color.White,
    background = CanvasIvory,
    onBackground = Ink,
    surface = SurfaceWhite,
    onSurface = Ink,
    surfaceVariant = Basil100,
    onSurfaceVariant = InkMuted,
    outline = LineSoft,
    error = ErrorRed,
    onError = Color.White,
)

private val NutritionShapes = Shapes(
    extraSmall = RoundedCornerShape(8.dp),
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(24.dp),
    extraLarge = RoundedCornerShape(32.dp),
)

@Composable
fun NutritionScannerTheme(content: @Composable () -> Unit) {
    NutritionSystemBars(color = CanvasIvory, useDarkIcons = true)
    MaterialTheme(
        colorScheme = NutritionLightColors,
        typography = NutritionTypography,
        shapes = NutritionShapes,
        content = content,
    )
}

/** Lets the live camera keep readable controls while document screens use light bars. */
@Composable
fun NutritionSystemBars(color: Color, useDarkIcons: Boolean) {
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = color.toArgb()
            window.navigationBarColor = color.toArgb()
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = useDarkIcons
                isAppearanceLightNavigationBars = useDarkIcons
            }
        }
    }
}
