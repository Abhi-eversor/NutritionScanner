# Add project specific ProGuard rules here.
# Keep TFLite classes
-keep class org.tensorflow.lite.** { *; }
# Keep ML Kit classes
-keep class com.google.mlkit.** { *; }
# Keep OkHttp
-dontwarn okhttp3.**
-keep class okhttp3.** { *; }
