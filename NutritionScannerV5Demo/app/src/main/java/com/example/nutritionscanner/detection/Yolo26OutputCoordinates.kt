package com.example.nutritionscanner.detection

/** Converts normalized or pixel-coordinate YOLO outputs to input pixels. */
internal object Yolo26OutputCoordinates {
    private const val NORMALIZED_COORDINATE_LIMIT = 2f

    fun pixelMultiplier(largestAbsoluteCoordinate: Float, inputSize: Int): Float =
        if (largestAbsoluteCoordinate <= NORMALIZED_COORDINATE_LIMIT) inputSize.toFloat() else 1f
}
