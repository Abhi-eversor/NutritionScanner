package com.example.nutritionscanner.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import com.example.nutritionscanner.detection.Detection
import com.example.nutritionscanner.theme.OverlayBoxGreen

/**
 * Draws bounding boxes over the camera preview for detected nutrition labels.
 */
@Composable
fun DetectionOverlay(
    detections: List<Detection>,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        for (detection in detections) {
            val box = detection.box
            val left = box.left * canvasWidth
            val top = box.top * canvasHeight
            val right = box.right * canvasWidth
            val bottom = box.bottom * canvasHeight
            val boxWidth = right - left
            val boxHeight = bottom - top

            drawRect(
                color = OverlayBoxGreen,
                topLeft = Offset(left, top),
                size = Size(boxWidth, boxHeight),
                style = Stroke(width = 4f),
            )

            val cornerLen = minOf(boxWidth, boxHeight) * 0.15f
            val cornerStroke = 8f

            drawLine(OverlayBoxGreen, Offset(left, top), Offset(left + cornerLen, top), cornerStroke)
            drawLine(OverlayBoxGreen, Offset(left, top), Offset(left, top + cornerLen), cornerStroke)
            drawLine(OverlayBoxGreen, Offset(right, top), Offset(right - cornerLen, top), cornerStroke)
            drawLine(OverlayBoxGreen, Offset(right, top), Offset(right, top + cornerLen), cornerStroke)
            drawLine(OverlayBoxGreen, Offset(left, bottom), Offset(left + cornerLen, bottom), cornerStroke)
            drawLine(OverlayBoxGreen, Offset(left, bottom), Offset(left, bottom - cornerLen), cornerStroke)
            drawLine(OverlayBoxGreen, Offset(right, bottom), Offset(right - cornerLen, bottom), cornerStroke)
            drawLine(OverlayBoxGreen, Offset(right, bottom), Offset(right, bottom - cornerLen), cornerStroke)

            val labelText = "Nutrition ${(detection.score * 100).toInt()}%"
            drawIntoCanvas { canvas ->
                val paint = android.graphics.Paint().apply {
                    color = 0xCC10B981.toInt() // Emerald with transparency
                    style = android.graphics.Paint.Style.FILL
                }
                val textPaint = android.graphics.Paint().apply {
                    color = android.graphics.Color.WHITE
                    textSize = 36f
                    isAntiAlias = true
                    typeface = android.graphics.Typeface.create(
                        android.graphics.Typeface.DEFAULT,
                        android.graphics.Typeface.BOLD
                    )
                }
                val textWidth = textPaint.measureText(labelText)
                val textHeight = 42f
                val padding = 12f

                canvas.nativeCanvas.drawRoundRect(
                    left,
                    top - textHeight - padding * 2,
                    left + textWidth + padding * 2,
                    top,
                    8f, 8f, paint,
                )
                canvas.nativeCanvas.drawText(
                    labelText,
                    left + padding,
                    top - padding - 6f,
                    textPaint,
                )
            }
        }
    }
}
