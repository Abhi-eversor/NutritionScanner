package com.example.nutritionscanner.ui

import android.graphics.Bitmap
import android.graphics.RectF
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import com.example.nutritionscanner.theme.OverlayBoxGreen
import kotlin.math.hypot

/** Crop editor that stores the selection in normalized image coordinates. */
@Composable
fun AdjustableCropPreview(
    bitmap: Bitmap,
    initialCrop: RectF,
    onCropChanged: (RectF) -> Unit,
    modifier: Modifier = Modifier,
) {
    var crop by remember(bitmap) { mutableStateOf(RectF(initialCrop)) }
    var containerSize by remember { mutableStateOf(IntSize.Zero) }
    var activeHandle by remember { mutableStateOf(CropHandle.NONE) }

    Box(
        modifier = modifier
            .clip(androidx.compose.material3.MaterialTheme.shapes.medium)
            .onSizeChanged { containerSize = it }
            .pointerInput(bitmap, containerSize) {
                detectDragGestures(
                    onDragStart = { point ->
                        val bounds = imageBounds(containerSize, bitmap)
                        activeHandle = cropHandleAt(point, crop, bounds)
                    },
                    onDragEnd = { activeHandle = CropHandle.NONE },
                    onDragCancel = { activeHandle = CropHandle.NONE },
                    onDrag = { change, dragAmount ->
                        val bounds = imageBounds(containerSize, bitmap)
                        if (activeHandle == CropHandle.NONE || bounds.width <= 0f || bounds.height <= 0f) {
                            return@detectDragGestures
                        }
                        change.consume()
                        crop = moveCrop(
                            crop = crop,
                            handle = activeHandle,
                            deltaX = dragAmount.x / bounds.width,
                            deltaY = dragAmount.y / bounds.height,
                        )
                        onCropChanged(RectF(crop))
                    },
                )
            },
    ) {
        Image(
            bitmap = bitmap.asImageBitmap(),
            contentDescription = "Adjust nutrition label crop",
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxSize(),
        )
        androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
            val bounds = imageBounds(IntSize(size.width.toInt(), size.height.toInt()), bitmap)
            if (bounds.width <= 0f || bounds.height <= 0f) return@Canvas

            val left = bounds.left + crop.left * bounds.width
            val top = bounds.top + crop.top * bounds.height
            val right = bounds.left + crop.right * bounds.width
            val bottom = bounds.top + crop.bottom * bounds.height
            val shade = Color.Black.copy(alpha = 0.48f)

            drawRect(shade, Offset(bounds.left, bounds.top), Size(left - bounds.left, bounds.height))
            drawRect(shade, Offset(right, bounds.top), Size(bounds.right - right, bounds.height))
            drawRect(shade, Offset(left, bounds.top), Size(right - left, top - bounds.top))
            drawRect(shade, Offset(left, bottom), Size(right - left, bounds.bottom - bottom))
            drawRect(
                color = OverlayBoxGreen,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                style = Stroke(width = 5f),
            )
            listOf(
                Offset(left, top),
                Offset(right, top),
                Offset(left, bottom),
                Offset(right, bottom),
            ).forEach { handle ->
                drawCircle(Color.White, radius = 14f, center = handle)
                drawCircle(OverlayBoxGreen, radius = 9f, center = handle)
            }
        }
    }
}

private enum class CropHandle { TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, MOVE, NONE }

private data class ImageBounds(val left: Float, val top: Float, val width: Float, val height: Float) {
    val right: Float get() = left + width
    val bottom: Float get() = top + height
}

private fun imageBounds(container: IntSize, bitmap: Bitmap): ImageBounds {
    if (container.width <= 0 || container.height <= 0 || bitmap.width <= 0 || bitmap.height <= 0) {
        return ImageBounds(0f, 0f, 0f, 0f)
    }
    val containerWidth = container.width.toFloat()
    val containerHeight = container.height.toFloat()
    val scale = minOf(containerWidth / bitmap.width, containerHeight / bitmap.height)
    val width = bitmap.width * scale
    val height = bitmap.height * scale
    return ImageBounds(
        left = (containerWidth - width) / 2f,
        top = (containerHeight - height) / 2f,
        width = width,
        height = height,
    )
}

private fun cropHandleAt(point: Offset, crop: RectF, bounds: ImageBounds): CropHandle {
    if (bounds.width <= 0f || bounds.height <= 0f) return CropHandle.NONE
    val left = bounds.left + crop.left * bounds.width
    val top = bounds.top + crop.top * bounds.height
    val right = bounds.left + crop.right * bounds.width
    val bottom = bounds.top + crop.bottom * bounds.height
    val radius = 42f
    fun near(x: Float, y: Float): Boolean = hypot(point.x - x, point.y - y) <= radius
    return when {
        near(left, top) -> CropHandle.TOP_LEFT
        near(right, top) -> CropHandle.TOP_RIGHT
        near(left, bottom) -> CropHandle.BOTTOM_LEFT
        near(right, bottom) -> CropHandle.BOTTOM_RIGHT
        point.x in left..right && point.y in top..bottom -> CropHandle.MOVE
        else -> CropHandle.NONE
    }
}

private fun moveCrop(crop: RectF, handle: CropHandle, deltaX: Float, deltaY: Float): RectF {
    val minSize = 0.08f
    return when (handle) {
        CropHandle.TOP_LEFT -> RectF(
            (crop.left + deltaX).coerceIn(0f, crop.right - minSize),
            (crop.top + deltaY).coerceIn(0f, crop.bottom - minSize),
            crop.right,
            crop.bottom,
        )
        CropHandle.TOP_RIGHT -> RectF(
            crop.left,
            (crop.top + deltaY).coerceIn(0f, crop.bottom - minSize),
            (crop.right + deltaX).coerceIn(crop.left + minSize, 1f),
            crop.bottom,
        )
        CropHandle.BOTTOM_LEFT -> RectF(
            (crop.left + deltaX).coerceIn(0f, crop.right - minSize),
            crop.top,
            crop.right,
            (crop.bottom + deltaY).coerceIn(crop.top + minSize, 1f),
        )
        CropHandle.BOTTOM_RIGHT -> RectF(
            crop.left,
            crop.top,
            (crop.right + deltaX).coerceIn(crop.left + minSize, 1f),
            (crop.bottom + deltaY).coerceIn(crop.top + minSize, 1f),
        )
        CropHandle.MOVE -> {
            val boundedX = deltaX.coerceIn(-crop.left, 1f - crop.right)
            val boundedY = deltaY.coerceIn(-crop.top, 1f - crop.bottom)
            RectF(crop.left + boundedX, crop.top + boundedY, crop.right + boundedX, crop.bottom + boundedY)
        }
        CropHandle.NONE -> RectF(crop)
    }
}
