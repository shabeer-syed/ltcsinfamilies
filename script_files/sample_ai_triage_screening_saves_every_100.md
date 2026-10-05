## Reproducible Sample Script
This script provides a self-contained, reproducible sample of the LLM clinical code triage pipeline configured on a validation set of 500 candidate EHR terms.
#**Note on Environment & Production Differences:**  
# This script is a demonstration harness and does not represent the exact infrastructure used during full-scale cohort derivation:
# - **Checkpointing:** Checkpoints every 100 items (the full production script processed ~75,000 codes and saved in batches of 1,000).
# - **Encoding & parsing:** Enhanced with `latin1` fallback decoding and strict string casting to prevent character-map errors and numeric ID truncation across different operating systems and Workbench setups.
# - **API & runtime:** Configured for the standalone Google GenAI SDK; production execution occurred within a secured Google Cloud Vertex AI Workbench running earlier runtime and model versions.

import json
import os
import time
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm

# 1. SETUP & AUTHENTICATION

PROJECT_ID = "INSERT"
LOCATION = "global"
MODEL_ID = "gemini-3.1-flash-lite"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# 2. LOAD DATA

print("Loading datasets...")

input_file = "test_500_codes.csv" #insert your codes
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file '{input_file}' not found.")

# FIX: encoding='latin1' bypasses byte 0x85; dtype ensures medcodes stay exact strings
df_target = pd.read_csv(input_file, encoding='latin1', dtype={'medcode': str})

required_cols = {'medcode', 'coding_term', 'indicator'}
missing_cols = required_cols - set(df_target.columns)
if missing_cols:
    raise ValueError(f"Input file {input_file} is missing required columns: {missing_cols}")

# Reference indicators: loads external reference or falls back to indicators in target dataset
ref_file = "INSERT_reference_indicator_dictionary.csv"
if os.path.exists(ref_file):
    df_ref = pd.read_csv(ref_file, encoding='latin1')
    valid_indicators = sorted(df_ref['indicator'].dropna().unique().tolist())
else:
    print(f"Notice: {ref_file} not found. Extracting reference indicators from {input_file}.")
    valid_indicators = sorted(df_target['indicator'].dropna().unique().tolist())

df_unique_codes = df_target[['medcode', 'coding_term', 'indicator']].drop_duplicates()
records = df_unique_codes.to_dict('records')
total_records = len(records)
print(f"Loaded {len(df_target)} total rows from {input_file} ({total_records} unique codes to classify).")

# 3. DEFINE STRICT JSON SCHEMA (Calibrated for High Sensitivity)
=
response_schema = {
    "type": "ARRAY",
    "description": "Array of high-sensitivity triage classifications corresponding exactly to the input batch.",
    "items": {
        "type": "OBJECT",
        "properties": {
            "medcode": {
                "type": "STRING",
                "description": "The exact medcode provided in the input record."
            },
            "decision": {
                "type": "STRING",
                "enum": ["INCLUDE", "EXCLUDE", "MAP_TO_OTHER", "CLINICAL_REVIEW"],
                "description": (
                    "INCLUDE: Valid or probable long-term condition appropriately matching the current indicator. "
                    "EXCLUDE: STRICTLY for codes that are OBVIOUSLY and UNAMBIGUOUSLY not long-term conditions (pure administrative, negative screen, self-limiting acute illness). "
                    "MAP_TO_OTHER: Valid or probable long-term condition requiring remapping to a more accurate indicator. "
                    "CLINICAL_REVIEW: Borderline, proxy, or ambiguous terms where chronicity cannot be ruled out."
                )
            },
            "target_indicator": {
                "type": "STRING",
                "description": (
                    "If MAP_TO_OTHER: The most specific clinical indicator. "
                    "If INCLUDE: Return the current indicator. "
                    "If EXCLUDE or CLINICAL_REVIEW: Return an empty string ''."
                )
            },
            "secondary_indicator": {
                "type": "STRING",
                "description": (
                    "Secondary condition if the term captures multiple distinct conditions; otherwise 'None'."
                )
            },
            "reasoning": {
                "type": "STRING",
                "description": "Short clinical rationale (max 10-12 words)."
            }
        },
        "required": ["medcode", "decision", "target_indicator", "secondary_indicator", "reasoning"]
    }
}


# 4. BATCHED SCREENING PIPELINE

API_BATCH_SIZE = 20
SAVE_CHUNK_SIZE = 100
results = []
last_saved_index = 0
file_counter = 1

for i in tqdm(range(0, total_records, API_BATCH_SIZE), desc=f"Triaging Codes (Batches of {API_BATCH_SIZE})"):
    current_batch = records[i : i + API_BATCH_SIZE]
    batch_lookup = {str(item['medcode']): item for item in current_batch}
    batch_json_str = json.dumps(current_batch, indent=2)

    prompt = f"""
You are an expert clinical epidemiologist triaging primary care electronic health record codes for a birth cohort study of long-term health conditions.
OPERATING PRINCIPLE: DELIBERATELY INCLUSIVE / HIGH-SENSITIVITY SCREENING
Your primary objective is to MINIMIZE FALSE NEGATIVES. Ascertainment must be inclusive:
- Admit close proxy markers of a condition and acute presentations carrying recognized risk of long-term sequelae.
- Inherent chronicity: Psychosis, bipolar disorder, cancer, diabetes, epilepsy, cerebral palsy, congenital anomalies, asthma, and chronic systemic illnesses.
- ALL mental health conditions, chronic symptoms, and psychiatric medications MUST be retained as long-term conditions.
- RULE FOR EXCLUSION: Apply 'EXCLUDE' ONLY if the code is OBVIOUSLY, BEYOND CLINICAL DOUBT, not a long-term condition (e.g., purely administrative invitations/did-not-attend, normal/negative screening results, family history only, or clearly minor transient conditions like common cold, acute gastroenteritis, or superficial abrasion).
- RULE FOR DOUBT: If you are uncertain whether a code represents a chronic condition, a proxy marker, or has long-term sequelae, DO NOT EXCLUDE IT. Classify it as 'INCLUDE', 'MAP_TO_OTHER', or escalate to 'CLINICAL_REVIEW'.

INPUT BATCH ({len(current_batch)} codes):
{batch_json_str}

TASK:
For EVERY item in the batch:
1. Assess chronicity under the HIGH-SENSITIVITY threshold:
   - ONLY set decision to 'EXCLUDE' if it is OBVIOUSLY not relevant or completely acute/administrative. Set target_indicator to '' and secondary_indicator to 'None'.
2. If it is or could plausibly be a long-term condition:
   - If current indicator is medically accurate: set decision to 'INCLUDE', target_indicator to current indicator, secondary_indicator to 'None'.
   - If clinically misplaced or too broad: set decision to 'MAP_TO_OTHER' and assign the most specific indicator from the Reference List below.
3. If borderline, ambiguous, or context-dependent: set decision to 'CLINICAL_REVIEW', target_indicator to '', secondary_indicator to 'None'.
4. MULTIPLE CONDITIONS: If the term represents multiple distinct conditions, assign the dominant/most severe to 'target_indicator' and the second to 'secondary_indicator'.
5. Provide a brief clinical 'reasoning' (max 10-12 words).

Reference List of Valid Indicators:
{valid_indicators}

CRITICAL: Return a JSON array containing exactly {len(current_batch)} objects matching the schema. Include the exact 'medcode' for each record.
"""

    max_retries = 4
    batch_success = False

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    )
                ),
            )

            output_array = json.loads(response.text)
            returned_medcodes = set()

            for output in output_array:
                returned_medcode = str(output.get('medcode'))
                if returned_medcode in batch_lookup:
                    returned_medcodes.add(returned_medcode)
                    orig_data = batch_lookup[returned_medcode]

                    if output.get('decision') == 'INCLUDE' and not output.get('target_indicator'):
                        output['target_indicator'] = orig_data['indicator']

                    output['medcode'] = str(orig_data['medcode'])
                    output['coding_term'] = orig_data['coding_term']
                    output['indicator'] = orig_data['indicator']

                    results.append(output)

            # High-sensitivity fallback: omitted codes go to CLINICAL_REVIEW rather than being dropped
            missing_in_batch = set(batch_lookup.keys()) - returned_medcodes
            for missing_code in missing_in_batch:
                orig_data = batch_lookup[missing_code]
                results.append({
                    'medcode': str(orig_data['medcode']),
                    'coding_term': orig_data['coding_term'],
                    'indicator': orig_data['indicator'],
                    'decision': 'CLINICAL_REVIEW',
                    'target_indicator': '',
                    'secondary_indicator': 'None',
                    'reasoning': 'API omitted code from batch array; routed for clinical review'
                })

            batch_success = True
            break

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "503" in error_msg:
                sleep_time = 2 ** (attempt + 1)
                tqdm.write(f"Rate/resource limit hit on batch {i}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                tqdm.write(f"Error on batch index {i} (attempt {attempt + 1}): {e}")
                time.sleep(1)

    if not batch_success:
        tqdm.write(f"Batch {i} failed after {max_retries} retries. Routing to CLINICAL_REVIEW.")
        for item in current_batch:
            results.append({
                'medcode': str(item['medcode']),
                'coding_term': item['coding_term'],
                'indicator': item['indicator'],
                'decision': 'CLINICAL_REVIEW',
                'target_indicator': '',
                'secondary_indicator': 'None',
                'reasoning': 'Batch request failure; routed for clinical review'
            })

    time.sleep(1.0)

    # Checkpoint logic
    if (len(results) - last_saved_index) >= SAVE_CHUNK_SIZE:
        df_chunk_ai = pd.DataFrame(results[last_saved_index:])
        df_chunk_merged = df_target.merge(
            df_chunk_ai,
            on=['medcode', 'coding_term', 'indicator'],
            how='inner'
        )
        checkpoint_filename = f"test_500_checkpoint_{file_counter}.csv"
        df_chunk_merged.to_csv(checkpoint_filename, index=False, encoding='utf-8')
        tqdm.write(f"Checkpoint saved: {checkpoint_filename}")
        file_counter += 1
        last_saved_index = len(results)

# =====================================================================
# 5. MERGE AND EXPORT FINAL RESULTS
# =====================================================================
print("\nExporting final results...")
df_all_ai = pd.DataFrame(results)

df_final = df_target.merge(
    df_all_ai[['medcode', 'coding_term', 'indicator', 'decision', 'target_indicator', 'secondary_indicator', 'reasoning']],
    on=['medcode', 'coding_term', 'indicator'],
    how='left'
)

output_filename = "test_500_codes_triaged.csv"
df_final.to_csv(output_filename, index=False, encoding='utf-8')

print(f"File successfully created: {output_filename}")
print("\nDecision summary:")
print(df_final['decision'].value_counts(dropna=False))
