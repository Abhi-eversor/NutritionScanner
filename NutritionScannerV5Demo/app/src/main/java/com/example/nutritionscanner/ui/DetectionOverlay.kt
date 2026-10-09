package com.example.nutritionscanner.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.drawscope.Stroke
import com.example.nutritionscanner.detection.Detection
import com.example.nutritionscanner.theme.OverlayBoxGreen

/** Draws normalized model boxes over the CameraX preview. */
@Composable
fun DetectionOverlay(
    detections: List<Detection>,
    sourceAspectRatio: Float,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        val sourceRatio = sourceAspectRatio.takeIf { it.isFinite() && it > 0f } ?: 3f / 4f
        val canvasRatio = size.width / size.height.coerceAtLeast(1f)
        val renderedWidth: Float
        val renderedHeight: Float
        val offsetX: Float
        val offsetY: Float
        if (canvasRatio > sourceRatio) {
            renderedWidth = size.width
            renderedHeight = renderedWidth / sourceRatio
            offsetX = 0f
            offsetY = (size.height - renderedHeight) / 2f
        } else {
            renderedHeight = size.height
            renderedWidth = renderedHeight * sourceRatio
            offsetX = (size.width - renderedWidth) / 2f
            offsetY = 0f
        }

        detections.forEach { detection ->
            val box = detection.box
            val left = offsetX + box.left.coerceIn(0f, 1f) * renderedWidth
            val top = offsetY + box.top.coerceIn(0f, 1f) * renderedHeight
            val right = offsetX + box.right.coerceIn(0f, 1f) * renderedWidth
            val bottom = offsetY + box.bottom.coerceIn(0f, 1f) * renderedHeight
            if (right <= left || bottom <= top) return@forEach

            drawRoundRect(
                color = OverlayBoxGreen,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                cornerRadius = CornerRadius(14f, 14f),
                style = Stroke(width = 5f),
            )
        }
    }
}
