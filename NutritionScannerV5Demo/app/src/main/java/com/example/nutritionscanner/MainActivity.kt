package com.example.nutritionscanner

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.nutritionscanner.theme.Basil100
import com.example.nutritionscanner.theme.Basil600
import com.example.nutritionscanner.theme.CanvasIvory
import com.example.nutritionscanner.theme.Ink
import com.example.nutritionscanner.theme.InkMuted
import com.example.nutritionscanner.theme.NutritionScannerTheme
import com.example.nutritionscanner.ui.CameraScreen
import com.example.nutritionscanner.ui.ResultsScreen

class MainActivity : ComponentActivity() {

    private val viewModel: ScannerViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        var hasCameraPermission by mutableStateOf(
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED,
        )
        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission(),
        ) { isGranted -> hasCameraPermission = isGranted }

        if (!hasCameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)

        setContent {
            NutritionScannerTheme {
                if (hasCameraPermission) {
                    NutritionScannerApp(viewModel)
                } else {
                    PermissionRequestScreen { permissionLauncher.launch(Manifest.permission.CAMERA) }
                }
            }
        }
    }
}

@Composable
fun NutritionScannerApp(viewModel: ScannerViewModel) {
    val scanState by viewModel.scanState.collectAsStateWithLifecycle()
    val healthProfile by viewModel.healthProfile.collectAsStateWithLifecycle()
    val detections by viewModel.detections.collectAsStateWithLifecycle()
    val fps by viewModel.fps.collectAsStateWithLifecycle()
    val liveFrameAspectRatio by viewModel.liveFrameAspectRatio.collectAsStateWithLifecycle()
    val isScanning = scanState is ScanState.Scanning

    Box(modifier = Modifier.fillMaxSize()) {
        CameraScreen(
            viewModel = viewModel,
            detections = if (isScanning) detections else emptyList(),
            fps = if (isScanning) fps else 0f,
            liveFrameAspectRatio = liveFrameAspectRatio,
        )
        AnimatedVisibility(
            visible = !isScanning,
            enter = slideInVertically(initialOffsetY = { it }) + fadeIn(),
            exit = slideOutVertically(targetOffsetY = { it }) + fadeOut(),
        ) {
            ResultsScreen(
                state = scanState,
                profile = healthProfile,
                onBack = viewModel::resetToScanning,
                onConfirmCrop = viewModel::confirmCrop,
                onRequestGuidance = viewModel::requestGuidance,
                onRetryRecommendation = viewModel::retryRecommendation,
                onSaveProfile = viewModel::saveProfile,
                onClearProfile = viewModel::clearProfile,
            )
        }
    }
}

@Composable
fun PermissionRequestScreen(onRequest: () -> Unit) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier.fillMaxSize().background(CanvasIvory),
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp),
        ) {
            Surface(color = Basil100, shape = MaterialTheme.shapes.large) {
                Icon(
                    Icons.Default.CameraAlt,
                    contentDescription = null,
                    tint = Basil600,
                    modifier = Modifier.padding(18.dp),
                )
            }
            Spacer(Modifier.height(20.dp))
            Text(
                "Ready to read your label",
                style = MaterialTheme.typography.headlineMedium,
                color = Ink,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Allow camera access to find a nutrition panel and help you crop it before reading.",
                style = MaterialTheme.typography.bodyMedium,
                color = InkMuted,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(24.dp))
            Button(
                onClick = onRequest,
                colors = ButtonDefaults.buttonColors(containerColor = Basil600),
            ) {
                Text("Allow camera")
            }
        }
    }
}
