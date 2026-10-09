package com.example.nutritionscanner

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.nutritionscanner.theme.*
import com.example.nutritionscanner.ui.CameraScreen
import com.example.nutritionscanner.ui.ResultsScreen

class MainActivity : ComponentActivity() {

    private val viewModel: ScannerViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        var hasCameraPermission by mutableStateOf(
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
        )

        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            hasCameraPermission = isGranted
        }

        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            NutritionScannerTheme {
                if (hasCameraPermission) {
                    NutritionScannerApp(viewModel)
                } else {
                    PermissionRequestScreen {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
            }
        }
    }
}

@Composable
fun NutritionScannerApp(viewModel: ScannerViewModel) {
    val scanState by viewModel.scanState.collectAsStateWithLifecycle()
    val detections by viewModel.detections.collectAsStateWithLifecycle()
    val fps by viewModel.fps.collectAsStateWithLifecycle()
    val healthProfile by viewModel.healthProfile.collectAsStateWithLifecycle()

    val isScanning = scanState is ScanState.Scanning

    Box(modifier = Modifier.fillMaxSize()) {
        CameraScreen(
            viewModel = viewModel,
            detections = if (isScanning) detections else emptyList(),
            fps = if (isScanning) fps else 0f,
        )

        AnimatedVisibility(
            visible = !isScanning,
            enter = slideInVertically(initialOffsetY = { it }) + fadeIn(),
            exit = slideOutVertically(targetOffsetY = { it }) + fadeOut(),
        ) {
            ResultsScreen(
                state = scanState,
                profile = healthProfile,
                onBack = { viewModel.resetToScanning() },
                onConfirmNutritionFacts = viewModel::confirmNutritionFacts,
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
        modifier = Modifier
            .fillMaxSize()
            .background(DarkSurface),
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(32.dp),
        ) {
            Text(
                "📷",
                style = MaterialTheme.typography.headlineLarge,
            )
            Spacer(Modifier.height(16.dp))
            Text(
                "Camera Permission Required",
                style = MaterialTheme.typography.headlineMedium,
                color = TextPrimary,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "This app needs camera access to scan nutrition labels in real-time.",
                style = MaterialTheme.typography.bodyMedium,
                color = TextSecondary,
                textAlign = TextAlign.Center,
            )
            Spacer(Modifier.height(24.dp))
            Button(
                onClick = onRequest,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Emerald500,
                ),
            ) {
                Text("Grant Camera Permission")
            }
        }
    }
}
