package com.example.nutritionscanner.ui

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.nutritionscanner.ScannerViewModel
import com.example.nutritionscanner.detection.Detection
import com.example.nutritionscanner.theme.DarkCard
import com.example.nutritionscanner.theme.Emerald400
import com.example.nutritionscanner.theme.Emerald500
import com.example.nutritionscanner.theme.Emerald600
import com.example.nutritionscanner.theme.ModerateYellow
import com.example.nutritionscanner.theme.OverlayBackground
import com.example.nutritionscanner.theme.TextMuted
import com.example.nutritionscanner.theme.TextPrimary
import com.example.nutritionscanner.theme.TextSecondary
import java.util.concurrent.Executors

/** Camera preview plus real-time detector overlay. */
@Composable
fun CameraScreen(
    viewModel: ScannerViewModel,
    detections: List<Detection>,
    fps: Float,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }

    DisposableEffect(Unit) {
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
                        return@addListener
                    }

                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }
                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                        .also { useCase ->
                            useCase.setAnalyzer(analysisExecutor, viewModel::processFrame)
                        }
                    val capture = ImageCapture.Builder()
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

        DetectionOverlay(detections = detections, modifier = Modifier.fillMaxSize())

        Surface(
            color = OverlayBackground,
            shape = MaterialTheme.shapes.small,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
                .statusBarsPadding(),
        ) {
            Text(
                text = "${fps.toInt()} FPS",
                color = if (fps >= 15f) Emerald400 else ModerateYellow,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            )
        }

        AnimatedVisibility(
            visible = detections.isNotEmpty(),
            enter = fadeIn() + scaleIn(),
            exit = fadeOut() + scaleOut(),
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(16.dp)
                .statusBarsPadding(),
        ) {
            Surface(color = Emerald600, shape = MaterialTheme.shapes.small) {
                Text(
                    text = "${detections.size} label${if (detections.size == 1) "" else "s"} found",
                    color = TextPrimary,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                )
            }
        }

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .background(OverlayBackground)
                .navigationBarsPadding()
                .padding(horizontal = 20.dp, vertical = 16.dp),
        ) {
            Text(
                text = if (detections.isNotEmpty()) {
                    "Nutrition panel detected — capture to scan"
                } else {
                    "Center the nutrition panel, then capture"
                },
                color = TextSecondary,
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(12.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(18.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = viewModel::openProfileEditor,
                    colors = ButtonDefaults.buttonColors(containerColor = DarkCard, contentColor = TextPrimary),
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
                                            viewModel.triggerScan(
                                                image.toBitmap().rotated(image.imageInfo.rotationDegrees),
                                            )
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
                    containerColor = if (detections.isNotEmpty()) Emerald500 else TextMuted,
                    contentColor = TextPrimary,
                    modifier = Modifier.size(72.dp),
                ) {
                    Icon(Icons.Default.CameraAlt, contentDescription = "Capture nutrition label", modifier = Modifier.size(32.dp))
                }
            }
        }
    }
}

private fun Bitmap.rotated(rotationDegrees: Int): Bitmap {
    if (rotationDegrees == 0) return this
    val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}
