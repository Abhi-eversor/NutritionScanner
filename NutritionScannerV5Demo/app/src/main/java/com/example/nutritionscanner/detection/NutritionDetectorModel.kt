package com.example.nutritionscanner.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.abs

/** A detected nutrition panel, expressed in normalized original-image coordinates. */
data class Detection(
    val box: RectF,
    val score: Float,
)

/**
 * Runtime for the quantized YOLO detector.
 * Input is [1, 3, 640, 640] NCHW RGB and output is [1, 5, 8400].
 */
class NutritionDetectorModel(context: Context) {

    companion object {
        private const val MODEL_FILE = "nutrition_label_yolo26s_int8.tflite"
        private const val INPUT_SIZE = 640
        private const val INPUT_CHANNELS = 3
        private const val OUTPUT_CHANNELS = 5
        private const val NUM_CANDIDATES = 8400
        private const val OUTPUT_INDEX = 0
        private const val SCORE_THRESHOLD = 0.49f
        private const val NMS_IOU_THRESHOLD = 0.70f
        private const val MAX_DETECTIONS = 20
        private const val MIN_NORMALIZED_BOX_AREA = 0.006f
        private const val LETTERBOX_COLOR = 114
        private const val INPUT_FLOATS = INPUT_SIZE * INPUT_SIZE * INPUT_CHANNELS
        private const val OUTPUT_FLOATS = OUTPUT_CHANNELS * NUM_CANDIDATES
    }

    private val interpreter = Interpreter(
        loadModelFile(context),
        Interpreter.Options().apply { numThreads = 4 },
    )
    private val inputBuffer = directFloatBuffer(INPUT_FLOATS)
    private val outputBuffer = directFloatBuffer(OUTPUT_FLOATS)
    private val outputValues = FloatArray(OUTPUT_FLOATS)
    private val inputPixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val inputTensor = interpreter.getInputTensor(0)
        require(inputTensor.shape().contentEquals(intArrayOf(1, INPUT_CHANNELS, INPUT_SIZE, INPUT_SIZE))) {
            "Expected YOLO26 input [1, 3, 640, 640], got ${inputTensor.shape().contentToString()}"
        }
        require(inputTensor.dataType() == DataType.FLOAT32) {
            "Expected FLOAT32 YOLO26 input, got ${inputTensor.dataType()}"
        }

        val outputTensor = interpreter.getOutputTensor(OUTPUT_INDEX)
        require(outputTensor.shape().contentEquals(intArrayOf(1, OUTPUT_CHANNELS, NUM_CANDIDATES))) {
            "Expected YOLO26 output [1, 5, 8400], got ${outputTensor.shape().contentToString()}"
        }
        require(outputTensor.dataType() == DataType.FLOAT32) {
            "Expected FLOAT32 YOLO26 output, got ${outputTensor.dataType()}"
        }
    }

    /** Runs letterboxed YOLO26 inference and class-agnostic NMS on a still image. */
    @Synchronized
    fun detect(bitmap: Bitmap): List<Detection> {
        require(bitmap.width > 0 && bitmap.height > 0) { "Cannot detect in an empty bitmap." }
        val transform = preprocess(bitmap)
        outputBuffer.clear()
        interpreter.runForMultipleInputsOutputs(
            arrayOf(inputBuffer.apply { rewind() }),
            mapOf(OUTPUT_INDEX to outputBuffer),
        )
        outputBuffer.rewind()
        for (index in outputValues.indices) outputValues[index] = outputBuffer.float
        return postProcess(transform)
    }

    @Synchronized
    fun close() = interpreter.close()

    /** Letterboxes an image and converts it to NCHW float input. */
    private fun preprocess(bitmap: Bitmap): LetterboxTransform {
        val scale = min(INPUT_SIZE.toFloat() / bitmap.width, INPUT_SIZE.toFloat() / bitmap.height)
        val resizedWidth = (bitmap.width * scale).roundToInt().coerceIn(1, INPUT_SIZE)
        val resizedHeight = (bitmap.height * scale).roundToInt().coerceIn(1, INPUT_SIZE)
        val horizontalPadding = INPUT_SIZE - resizedWidth
        val verticalPadding = INPUT_SIZE - resizedHeight
        // Ultralytics' +/- 0.1 convention makes odd padding deterministic.
        val left = (horizontalPadding / 2f - 0.1f).roundToInt()
        val top = (verticalPadding / 2f - 0.1f).roundToInt()

        val letterboxed = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        try {
            Canvas(letterboxed).apply {
                drawColor(Color.rgb(LETTERBOX_COLOR, LETTERBOX_COLOR, LETTERBOX_COLOR))
                drawBitmap(
                    bitmap,
                    null,
                    Rect(left, top, left + resizedWidth, top + resizedHeight),
                    bitmapPaint,
                )
            }
            letterboxed.getPixels(inputPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        } finally {
            letterboxed.recycle()
        }

        inputBuffer.clear()
        // The exported model is NCHW, so write each full color plane in turn.
        for (channel in 0 until INPUT_CHANNELS) {
            for (pixel in inputPixels) {
                val value = when (channel) {
                    0 -> (pixel shr 16) and 0xff
                    1 -> (pixel shr 8) and 0xff
                    else -> pixel and 0xff
                }
                inputBuffer.putFloat(value / 255f)
            }
        }
        inputBuffer.rewind()

        return LetterboxTransform(
            sourceWidth = bitmap.width,
            sourceHeight = bitmap.height,
            scale = scale,
            padLeft = left.toFloat(),
            padTop = top.toFloat(),
        )
    }

    /** Decodes channel-major YOLO output [cx, cy, w, h, confidence] and runs NMS. */
    private fun postProcess(transform: LetterboxTransform): List<Detection> {
        val candidates = ArrayList<Detection>()
        val coordinateMultiplier = Yolo26OutputCoordinates.pixelMultiplier(
            largestAbsoluteCoordinate = largestDetectedCoordinate(),
            inputSize = INPUT_SIZE,
        )
        for (candidateIndex in 0 until NUM_CANDIDATES) {
            val score = outputValues[4 * NUM_CANDIDATES + candidateIndex]
            if (!score.isFinite() || score < SCORE_THRESHOLD) continue

            val centerX = outputValues[candidateIndex]
            val centerY = outputValues[NUM_CANDIDATES + candidateIndex]
            val width = outputValues[2 * NUM_CANDIDATES + candidateIndex]
            val height = outputValues[3 * NUM_CANDIDATES + candidateIndex]
            if (!centerX.isFinite() || !centerY.isFinite() || !width.isFinite() || !height.isFinite() ||
                width <= 0f || height <= 0f
            ) continue

            // This model exposes normalized [0, 1] coordinates. Convert them to
            // 640px letterbox coordinates before undoing padding and scale. The
            // adaptive multiplier also supports a future pixel-coordinate export.
            val box = transform.toNormalizedBox(
                centerX * coordinateMultiplier,
                centerY * coordinateMultiplier,
                width * coordinateMultiplier,
                height * coordinateMultiplier,
            ) ?: continue
            if (box.width() * box.height() < MIN_NORMALIZED_BOX_AREA) continue
            candidates += Detection(box = box, score = score.coerceAtMost(1f))
        }
        return nonMaximumSuppression(candidates)
    }

    /** Finds the coordinate convention only among candidates the detector trusts. */
    private fun largestDetectedCoordinate(): Float {
        var largest = 0f
        for (candidateIndex in 0 until NUM_CANDIDATES) {
            val score = outputValues[4 * NUM_CANDIDATES + candidateIndex]
            if (!score.isFinite() || score < SCORE_THRESHOLD) continue
            for (channel in 0 until 4) {
                val value = outputValues[channel * NUM_CANDIDATES + candidateIndex]
                if (value.isFinite()) largest = max(largest, abs(value))
            }
        }
        return largest
    }

    private fun nonMaximumSuppression(candidates: List<Detection>): List<Detection> {
        val kept = ArrayList<Detection>(MAX_DETECTIONS)
        for (candidate in candidates.sortedByDescending { it.score }) {
            if (kept.none { existing -> iou(existing.box, candidate.box) > NMS_IOU_THRESHOLD }) {
                kept += candidate
                if (kept.size == MAX_DETECTIONS) break
            }
        }
        return kept
    }

    private fun iou(first: RectF, second: RectF): Float {
        val left = max(first.left, second.left)
        val top = max(first.top, second.top)
        val right = min(first.right, second.right)
        val bottom = min(first.bottom, second.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = first.width() * first.height() + second.width() * second.height() - intersection
        return intersection / max(union, 1e-7f)
    }

    private fun directFloatBuffer(floatCount: Int): ByteBuffer =
        ByteBuffer.allocateDirect(floatCount * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())

    private fun loadModelFile(context: Context): MappedByteBuffer =
        context.assets.openFd(MODEL_FILE).use { asset ->
            FileInputStream(asset.fileDescriptor).channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, asset.startOffset, asset.declaredLength)
            }
        }

    private data class LetterboxTransform(
        val sourceWidth: Int,
        val sourceHeight: Int,
        val scale: Float,
        val padLeft: Float,
        val padTop: Float,
    ) {
        fun toNormalizedBox(centerX: Float, centerY: Float, width: Float, height: Float): RectF? {
            val left = ((centerX - width / 2f - padLeft) / scale).coerceIn(0f, sourceWidth.toFloat())
            val top = ((centerY - height / 2f - padTop) / scale).coerceIn(0f, sourceHeight.toFloat())
            val right = ((centerX + width / 2f - padLeft) / scale).coerceIn(0f, sourceWidth.toFloat())
            val bottom = ((centerY + height / 2f - padTop) / scale).coerceIn(0f, sourceHeight.toFloat())
            if (right <= left || bottom <= top) return null
            return RectF(
                left / sourceWidth,
                top / sourceHeight,
                right / sourceWidth,
                bottom / sourceHeight,
            )
        }
    }
}
