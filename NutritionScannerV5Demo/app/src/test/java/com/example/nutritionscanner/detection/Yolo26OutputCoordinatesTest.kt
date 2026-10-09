package com.example.nutritionscanner.detection

import org.junit.Assert.assertEquals
import org.junit.Test

class Yolo26OutputCoordinatesTest {

    @Test
    fun `normalised YOLO boxes are converted to input pixels`() {
        assertEquals(640f, Yolo26OutputCoordinates.pixelMultiplier(0.92f, 640), 0f)
    }

    @Test
    fun `pixel-coordinate YOLO boxes remain unchanged`() {
        assertEquals(1f, Yolo26OutputCoordinates.pixelMultiplier(241f, 640), 0f)
    }
}
