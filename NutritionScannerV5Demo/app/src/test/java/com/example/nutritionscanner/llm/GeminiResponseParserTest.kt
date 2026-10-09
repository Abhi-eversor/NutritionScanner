package com.example.nutritionscanner.llm

import com.example.nutritionscanner.profile.HealthProfile
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GeminiResponseParserTest {

    @Test
    fun `accepts ordinary free-text advice without a schema`() {
        val result = GeminiResponseParser.parseApiResponse(
            """{"candidates":[{"content":{"parts":[{"text":"This label is relatively high in sodium. Compare the serving size with your usual portion."}]}}]}""",
        )

        assertTrue(result is GeminiResponseParser.EnvelopeResult.Success)
        assertEquals(
            "This label is relatively high in sodium. Compare the serving size with your usual portion.",
            (result as GeminiResponseParser.EnvelopeResult.Success).advice.text,
        )
    }

    @Test
    fun `joins multiple Gemini text parts while preserving a readable break`() {
        val result = GeminiResponseParser.parseApiResponse(
            """{"candidates":[{"content":{"parts":[{"text":"First paragraph."},{"text":"Second paragraph."}]}}]}""",
        ) as GeminiResponseParser.EnvelopeResult.Success

        assertEquals("First paragraph.\nSecond paragraph.", result.advice.text)
    }

    @Test
    fun `removes only surrounding code fences from otherwise plain advice`() {
        val result = GeminiResponseParser.parseApiResponse(
            """{"candidates":[{"content":{"parts":[{"text":"```\nUse the serving size as your comparison point.\n```"}]}}]}""",
        ) as GeminiResponseParser.EnvelopeResult.Success

        assertEquals("Use the serving size as your comparison point.", result.advice.text)
    }

    @Test
    fun `returns an empty response failure when candidates contain no text`() {
        val result = GeminiResponseParser.parseApiResponse("""{"candidates":[{"content":{"parts":[]}}]}""")

        assertTrue(result is GeminiResponseParser.EnvelopeResult.Failure)
        assertEquals(
            GeminiFailureReason.EMPTY_RESPONSE,
            (result as GeminiResponseParser.EnvelopeResult.Failure).reason,
        )
    }

    @Test
    fun `returns a blocked failure for safety candidates`() {
        val result = GeminiResponseParser.parseApiResponse(
            """{"candidates":[{"finishReason":"SAFETY","content":{"parts":[]}}]}""",
        )

        assertTrue(result is GeminiResponseParser.EnvelopeResult.Failure)
        assertEquals(
            GeminiFailureReason.BLOCKED,
            (result as GeminiResponseParser.EnvelopeResult.Failure).reason,
        )
    }

    @Test
    fun `returns invalid response failure for a malformed envelope`() {
        val result = GeminiResponseParser.parseApiResponse("not json")

        assertTrue(result is GeminiResponseParser.EnvelopeResult.Failure)
        assertEquals(
            GeminiFailureReason.INVALID_RESPONSE,
            (result as GeminiResponseParser.EnvelopeResult.Failure).reason,
        )
    }

    @Test
    fun `request contains reviewed raw text and profile but no response schema`() {
        val client = GeminiClient()
        try {
            val request = client.buildRequestBody(
                reviewedLabelText = "Energy 531 kcal\nSodium 677 mg",
                originalOcrText = "Energy 531 kcal\nSodium 677 mg\nPer 100 g",
                sourceLanguage = "en",
                profile = HealthProfile(hasDiabetes = true, reducedSodium = true),
                responseLanguage = "English",
            )
            val prompt = Json.parseToJsonElement(request)
                .jsonObject["contents"]!!
                .jsonArray[0]
                .jsonObject["parts"]!!
                .jsonArray[0]
                .jsonObject["text"]!!
                .jsonPrimitive.content

            assertTrue(prompt.contains("USER_REVIEWED_LABEL_TEXT"))
            assertTrue(prompt.contains("Energy 531 kcal"))
            assertTrue(prompt.contains("ORIGINAL_OCR_TEXT"))
            assertTrue(prompt.contains("hasDiabetes"))
            assertFalse(request.contains("responseJsonSchema"))
            assertFalse(request.contains("responseMimeType"))
            assertFalse(request.contains("caloriesKcal"))
        } finally {
            client.close()
        }
    }
}
