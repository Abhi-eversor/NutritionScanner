package com.example.nutritionscanner.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** A detected nutrition panel, expressed in normalized image coordinates. */
data class Detection(
    val box: RectF,
    val score: Float,
)

/**
 * Runtime for the quantized MobileNetV2 SSD model.
 * Output tensors are [1, 510, 4] encoded boxes and [1, 510, 2] class logits.
 */
class NutritionDetectorModel(context: Context) {

    companion object {
        private const val MODEL_FILE = "nutrition_detector.tflite"
        private const val IMAGE_SIZE = 128
        private const val NUM_CLASSES = 2
        private const val NUM_ANCHORS = 510
        private const val BOX_OUTPUT_INDEX = 0
        private const val CLASS_OUTPUT_INDEX = 1

        private val FEATURE_MAP_SIZES = intArrayOf(8, 4, 2, 1)
        private val ASPECT_RATIOS = floatArrayOf(1f, 2f, 0.5f, 3f, 1f / 3f)
        private const val ANCHOR_SCALE_MIN = 0.2f
        private const val ANCHOR_SCALE_MAX = 0.9f
        private val BOX_VARIANCE = floatArrayOf(0.1f, 0.1f, 0.2f, 0.2f)

        /** Chosen from the external-test operating-point sweep (best F1 near 0.30). */
        private const val SCORE_THRESHOLD = 0.30f
        private const val NMS_IOU_THRESHOLD = 0.45f
        private const val MAX_DETECTIONS = 20
        private const val INPUT_BYTES = IMAGE_SIZE * IMAGE_SIZE * 3
    }

    private val interpreter = Interpreter(
        loadModelFile(context),
        Interpreter.Options().apply { numThreads = 4 },
    )
    private val anchors = generateAnchors()
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val boxScale: Float
    private val boxZeroPoint: Int
    private val classScale: Float
    private val classZeroPoint: Int

    init {
        require(anchors.size == NUM_ANCHORS) {
            "Anchor configuration produced ${anchors.size}; v5 requires $NUM_ANCHORS."
        }

        val inputTensor = interpreter.getInputTensor(0)
        require(inputTensor.shape().contentEquals(intArrayOf(1, IMAGE_SIZE, IMAGE_SIZE, 3))) {
            "Expected v5 input [1, 128, 128, 3], got ${inputTensor.shape().contentToString()}"
        }
        require(inputTensor.dataType().name == "UINT8") {
            "Expected a UINT8 v5 model input, got ${inputTensor.dataType()}"
        }
        inputTensor.quantizationParams().also {
            inputScale = it.scale
            inputZeroPoint = it.zeroPoint
        }

        // Materialize dynamic output tensors and validate the bundled model.
        runWarmUpInference()
        val boxTensor = interpreter.getOutputTensor(BOX_OUTPUT_INDEX)
        val classTensor = interpreter.getOutputTensor(CLASS_OUTPUT_INDEX)
        require(boxTensor.shape().contentEquals(intArrayOf(1, NUM_ANCHORS, 4))) {
            "Expected encoded boxes [1, 510, 4], got ${boxTensor.shape().contentToString()}"
        }
        require(classTensor.shape().contentEquals(intArrayOf(1, NUM_ANCHORS, NUM_CLASSES))) {
            "Expected class logits [1, 510, 2], got ${classTensor.shape().contentToString()}"
        }
        require(boxTensor.dataType().name == "UINT8" && classTensor.dataType().name == "UINT8") {
            "The v5 detector must have UINT8 output tensors."
        }
        boxTensor.quantizationParams().also {
            boxScale = it.scale
            boxZeroPoint = it.zeroPoint
        }
        classTensor.quantizationParams().also {
            classScale = it.scale
            classZeroPoint = it.zeroPoint
        }
    }

    /** Runs quantized inference, decodes SSD offsets, and applies class-aware NMS. */
    @Synchronized
    fun detect(bitmap: Bitmap): List<Detection> {
        val input = preprocess(bitmap)
        val boxBuffer = directBuffer(NUM_ANCHORS * 4)
        val classBuffer = directBuffer(NUM_ANCHORS * NUM_CLASSES)
        interpreter.runForMultipleInputsOutputs(
            arrayOf(input),
            mapOf(
                BOX_OUTPUT_INDEX to boxBuffer,
                CLASS_OUTPUT_INDEX to classBuffer,
            ),
        )

        val encodedBoxes = dequantize(boxBuffer, NUM_ANCHORS * 4, boxScale, boxZeroPoint)
        val logits = dequantize(classBuffer, NUM_ANCHORS * NUM_CLASSES, classScale, classZeroPoint)
        return postProcess(encodedBoxes, logits)
    }

    fun close() = interpreter.close()

    /** Mirrors training preprocessing: resize -> (RGB / 127.5) - 1 -> UINT8 quantization. */
    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val resized = if (bitmap.width == IMAGE_SIZE && bitmap.height == IMAGE_SIZE) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        }
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        return directBuffer(INPUT_BYTES).apply {
            for (pixel in pixels) {
                put(quantizeNormalized((pixel shr 16) and 0xff))
                put(quantizeNormalized((pixel shr 8) and 0xff))
                put(quantizeNormalized(pixel and 0xff))
            }
            rewind()
        }
    }

    private fun quantizeNormalized(channel: Int): Byte {
        val normalized = channel / 127.5f - 1f
        val quantized = (normalized / inputScale + inputZeroPoint)
            .roundToInt()
            .coerceIn(0, 255)
        return quantized.toByte()
    }

    private fun runWarmUpInference() {
        val input = directBuffer(INPUT_BYTES).apply {
            repeat(INPUT_BYTES) { put(quantizeNormalized(0)) }
            rewind()
        }
        interpreter.runForMultipleInputsOutputs(
            arrayOf(input),
            mapOf(
                BOX_OUTPUT_INDEX to directBuffer(NUM_ANCHORS * 4),
                CLASS_OUTPUT_INDEX to directBuffer(NUM_ANCHORS * NUM_CLASSES),
            ),
        )
    }

    private fun dequantize(
        buffer: ByteBuffer,
        count: Int,
        scale: Float,
        zeroPoint: Int,
    ): FloatArray {
        buffer.rewind()
        return FloatArray(count) {
            ((buffer.get().toInt() and 0xff) - zeroPoint) * scale
        }
    }

    private fun generateAnchors(): Array<FloatArray> {
        val scales = FloatArray(FEATURE_MAP_SIZES.size) { index ->
            ANCHOR_SCALE_MIN +
                (ANCHOR_SCALE_MAX - ANCHOR_SCALE_MIN) * index / (FEATURE_MAP_SIZES.size - 1)
        }
        val anchors = ArrayList<FloatArray>(NUM_ANCHORS)

        FEATURE_MAP_SIZES.forEachIndexed { level, featureMapSize ->
            val scale = scales[level]
            val nextScale = if (level == scales.lastIndex) 1.05f else scales[level + 1]
            val extraScale = sqrt(scale * nextScale)
            for (row in 0 until featureMapSize) {
                val cy = (row + 0.5f) / featureMapSize
                for (column in 0 until featureMapSize) {
                    val cx = (column + 0.5f) / featureMapSize
                    anchors += floatArrayOf(cx, cy, scale, scale)
                    anchors += floatArrayOf(cx, cy, extraScale, extraScale)
                    ASPECT_RATIOS.filter { it != 1f }.forEach { aspectRatio ->
                        anchors += floatArrayOf(
                            cx,
                            cy,
                            scale * sqrt(aspectRatio),
                            scale / sqrt(aspectRatio),
                        )
                    }
                }
            }
        }
        return anchors.toTypedArray()
    }

    private fun postProcess(encodedBoxes: FloatArray, logits: FloatArray): List<Detection> {
        val candidates = ArrayList<Detection>()
        for (anchorIndex in 0 until NUM_ANCHORS) {
            val classOffset = anchorIndex * NUM_CLASSES
            val backgroundLogit = logits[classOffset]
            val foregroundLogit = logits[classOffset + 1]
            val maxLogit = max(backgroundLogit, foregroundLogit)
            val background = exp((backgroundLogit - maxLogit).toDouble()).toFloat()
            val foreground = exp((foregroundLogit - maxLogit).toDouble()).toFloat()
            val score = foreground / (background + foreground)
            if (score < SCORE_THRESHOLD) continue

            val anchor = anchors[anchorIndex]
            val boxOffset = anchorIndex * 4
            val centerX = encodedBoxes[boxOffset] * BOX_VARIANCE[0] * anchor[2] + anchor[0]
            val centerY = encodedBoxes[boxOffset + 1] * BOX_VARIANCE[1] * anchor[3] + anchor[1]
            val width = exp((encodedBoxes[boxOffset + 2] * BOX_VARIANCE[2]).toDouble()).toFloat() * anchor[2]
            val height = exp((encodedBoxes[boxOffset + 3] * BOX_VARIANCE[3]).toDouble()).toFloat() * anchor[3]

            val box = RectF(
                (centerX - width / 2f).coerceIn(0f, 1f),
                (centerY - height / 2f).coerceIn(0f, 1f),
                (centerX + width / 2f).coerceIn(0f, 1f),
                (centerY + height / 2f).coerceIn(0f, 1f),
            )
            if (box.width() > 0f && box.height() > 0f) candidates += Detection(box, score)
        }
        return nonMaximumSuppression(candidates)
    }

    private fun nonMaximumSuppression(candidates: List<Detection>): List<Detection> {
        val sorted = candidates.sortedByDescending { it.score }
        val kept = ArrayList<Detection>(MAX_DETECTIONS)
        for (candidate in sorted) {
            if (kept.none { iou(it.box, candidate.box) > NMS_IOU_THRESHOLD }) {
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

    private fun directBuffer(bytes: Int): ByteBuffer =
        ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

    private fun loadModelFile(context: Context): MappedByteBuffer =
        context.assets.openFd(MODEL_FILE).use { asset ->
            FileInputStream(asset.fileDescriptor).channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, asset.startOffset, asset.declaredLength)
            }
        }
}
