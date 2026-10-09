package com.example.nutritionscanner.profile

import android.content.Context
import androidx.datastore.preferences.core.PreferenceDataStoreFactory
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.stringSetPreferencesKey
import androidx.datastore.preferences.preferencesDataStoreFile
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/** Persists only the selected profile flags; scans, photos, and OCR stay in memory. */
class HealthProfileRepository(context: Context) {
    private val dataStore = PreferenceDataStoreFactory.create {
        context.preferencesDataStoreFile("health_profile.preferences_pb")
    }

    val profile: Flow<HealthProfile> = dataStore.data.map { preferences ->
        HealthProfile(
            dietaryPreference = preferences[DIETARY_PREFERENCE]
                ?.let { value -> DietaryPreference.entries.firstOrNull { it.name == value } }
                ?: DietaryPreference.NONE,
            allergies = preferences[ALLERGIES].orEmpty().toList().sorted(),
            hasDiabetes = preferences[DIABETES] ?: false,
            hasHypertension = preferences[HYPERTENSION] ?: false,
            hasKidneyDisease = preferences[KIDNEY_DISEASE] ?: false,
            isPregnant = preferences[PREGNANCY] ?: false,
            reducedSugar = preferences[REDUCED_SUGAR] ?: false,
            reducedSodium = preferences[REDUCED_SODIUM] ?: false,
        )
    }

    suspend fun save(profile: HealthProfile) {
        dataStore.edit { preferences ->
            preferences[DIETARY_PREFERENCE] = profile.dietaryPreference.name
            preferences[ALLERGIES] = profile.allergies
                .map(String::trim)
                .filter(String::isNotBlank)
                .toSet()
            preferences[DIABETES] = profile.hasDiabetes
            preferences[HYPERTENSION] = profile.hasHypertension
            preferences[KIDNEY_DISEASE] = profile.hasKidneyDisease
            preferences[PREGNANCY] = profile.isPregnant
            preferences[REDUCED_SUGAR] = profile.reducedSugar
            preferences[REDUCED_SODIUM] = profile.reducedSodium
        }
    }

    suspend fun clear() = dataStore.edit { it.clear() }

    private companion object {
        val DIETARY_PREFERENCE = stringPreferencesKey("dietary_preference")
        val ALLERGIES = stringSetPreferencesKey("allergies")
        val DIABETES = booleanPreferencesKey("diabetes")
        val HYPERTENSION = booleanPreferencesKey("hypertension")
        val KIDNEY_DISEASE = booleanPreferencesKey("kidney_disease")
        val PREGNANCY = booleanPreferencesKey("pregnancy")
        val REDUCED_SUGAR = booleanPreferencesKey("reduced_sugar")
        val REDUCED_SODIUM = booleanPreferencesKey("reduced_sodium")
    }
}
