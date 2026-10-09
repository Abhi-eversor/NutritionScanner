package com.example.nutritionscanner.ui

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import android.util.Size
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.nutritionscanner.ScannerViewModel
import com.example.nutritionscanner.detection.Detection
import com.example.nutritionscanner.theme.Basil600
import com.example.nutritionscanner.theme.Ink
import com.example.nutritionscanner.theme.InkMuted
import com.example.nutritionscanner.theme.NutritionSystemBars
import com.example.nutritionscanner.theme.SurfaceWhite
import java.util.concurrent.Executors

/** Camera preview with a live detector overlay. */
@Composable
fun CameraScreen(
    viewModel: ScannerViewModel,
    detections: List<Detection>,
    fps: Float,
    liveFrameAspectRatio: Float,
    modifier: Modifier = Modifier,
) {
    NutritionSystemBars(color = Color.Black, useDarkIcons = false)
    val lifecycleOwner = LocalLifecycleOwner.current
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }

    DisposableEffect(analysisExecutor, captureExecutor) {
        onDispose {
            analysisExecutor.shutdown()
            captureExecutor.shutdown()
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        AndroidView(
            factory = { viewContext ->
                val previewView = PreviewView(viewContext).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                }
                val cameraProviderFuture = ProcessCameraProvider.getInstance(viewContext)
                cameraProviderFuture.addListener({
                    val cameraProvider = try {
                        cameraProviderFuture.get()
                    } catch (error: Exception) {
                        Log.e("CameraScreen", "Unable to obtain camera provider", error)
                        viewModel.reportCaptureError()
                        return@addListener
                    }

                    val preview = Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build()
                        .also { it.surfaceProvider = previewView.surfaceProvider }
                    val analysis = ImageAnalysis.Builder()
                        .setTargetResolution(Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                        .also { imageAnalysis ->
                            imageAnalysis.setAnalyzer(analysisExecutor, viewModel::processFrame)
                        }
                    val capture = ImageCapture.Builder()
                        // Cap still captures to keep on-device processing responsive.
                        .setTargetResolution(Size(1280, 960))
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build()

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            analysis,
                            capture,
                        )
                        imageCapture = capture
                    } catch (error: Exception) {
                        Log.e("CameraScreen", "Camera bind failed", error)
                        viewModel.reportCaptureError()
                    }
                }, ContextCompat.getMainExecutor(viewContext))
                previewView
            },
            modifier = Modifier.fillMaxSize(),
        )

        DetectionOverlay(
            detections = detections,
            sourceAspectRatio = liveFrameAspectRatio,
            modifier = Modifier.fillMaxSize(),
        )

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(16.dp)
                .statusBarsPadding(),
        ) {
            Surface(
                color = SurfaceWhite.copy(alpha = 0.94f),
                shape = MaterialTheme.shapes.small,
            ) {
                Text(
                    text = if (fps >= 1f) "Scanning · ${fps.toInt()} FPS" else "Scanning label",
                    color = Ink,
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                )
            }
            if (detections.isNotEmpty()) {
                Surface(
                    color = Basil600,
                    shape = MaterialTheme.shapes.small,
                ) {
                    Text(
                        text = "Label found",
                        color = Color.White,
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    )
                }
            }
        }

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(SurfaceWhite.copy(alpha = 0.96f))
                .navigationBarsPadding()
                .padding(horizontal = 20.dp, vertical = 16.dp),
        ) {
            Text(
                text = if (detections.isEmpty()) {
                    "Center the nutrition panel in the guide, then capture it."
                } else {
                    "Label found. Capture it, then fine-tune the crop before OCR."
                },
                color = InkMuted,
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(12.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(18.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = viewModel::openProfileEditor,
                    colors = ButtonDefaults.buttonColors(containerColor = Color.White, contentColor = Ink),
                ) {
                    Text("Profile")
                }
                FloatingActionButton(
                    onClick = {
                        val capture = imageCapture
                        if (capture == null) {
                            viewModel.reportCaptureError()
                        } else {
                            capture.takePicture(
                                captureExecutor,
                                object : ImageCapture.OnImageCapturedCallback() {
                                    override fun onCaptureSuccess(image: ImageProxy) {
                                        try {
                                            viewModel.triggerScan(image.toUprightBitmap())
                                        } catch (error: Exception) {
                                            Log.e("CameraScreen", "Photo conversion failed", error)
                                            viewModel.reportCaptureError()
                                        } finally {
                                            image.close()
                                        }
                                    }

                                    override fun onError(exception: ImageCaptureException) {
                                        Log.e("CameraScreen", "Photo capture failed", exception)
                                        viewModel.reportCaptureError()
                                    }
                                },
                            )
                        }
                    },
                    containerColor = Basil600,
                    contentColor = Color.White,
                    modifier = Modifier.size(72.dp),
                ) {
                    Icon(
                        Icons.Default.CameraAlt,
                        contentDescription = "Capture nutrition label",
                        modifier = Modifier.size(32.dp),
                    )
                }
            }
        }
    }
}

private fun ImageProxy.toUprightBitmap(): Bitmap {
    val rawBitmap = toBitmap()
    val uprightBitmap = rawBitmap.rotated(imageInfo.rotationDegrees)
    if (uprightBitmap !== rawBitmap) rawBitmap.recycle()
    return uprightBitmap
}

private fun Bitmap.rotated(rotationDegrees: Int): Bitmap {
    if (rotationDegrees == 0) return this
    val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}
