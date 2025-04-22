import pandas as pd
import numpy as np

# ensure pandas shows all columns when printing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD METADATA & SELECT RELEVANT COLUMNS
#    We start by reading the clinician‐curated Excel sheet and
#    keeping only the session identifier, the time of visit,
#    and the raw diagnosis string (dx1).
# ─────────────────────────────────────────────────────────────────────────────
meta_df = pd.read_excel('data/connectome_pipeline/meta_data.xlsx', engine='openpyxl')
meta_df = meta_df[['OASIS_session_label', 'days_to_visit', 'dx1']]
print(f"1) Loaded {len(meta_df)} total visits from metadata")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MAP TEXT DIAGNOSES → NUMERIC CATEGORIES
#    Collapse the many dx1 strings into three clinically‐meaningful bins:
#      0 = Healthy Control (HC)
#      1 = Mild Cognitive Impairment (MCI / uncertain)
#      2 = Alzheimer’s Dementia (AD / DAT / any 'AD dem*')
#
#    Rows mapping to NaN are dropped because they’re either
#    non‑AD dementias we’re excluding or placeholder entries.
# ─────────────────────────────────────────────────────────────────────────────
def map_dx1_to_label(dx1):
    if dx1 in ('Cognitively normal', 'No dementia'):
        return 0
    mci_set = {
        'uncertain dementia','Unc: ques. Impairment',
        'uncertain, possible NON AD dem','Unc: impair reversible',
        '0.5 in memory only','Incipient Non-AD dem',
        'Incipient demt PTP','ProAph w/o dement'
    }
    if dx1 in mci_set:
        return 1
    if dx1 == 'AD Dementia' or dx1 == 'DAT' or 'AD dem' in dx1 or dx1.startswith('DAT'):
        return 2
    return np.nan

meta_df['diagnosis_label'] = meta_df['dx1'].apply(map_dx1_to_label)
meta_df.dropna(subset=['diagnosis_label'], inplace=True)
meta_df['diagnosis_label'] = meta_df['diagnosis_label'].astype(int)
print(f"2) {len(meta_df)} visits remain after mapping to {{0,1,2}} labels")

# ─────────────────────────────────────────────────────────────────────────────
# 3. STANDARDIZE & SPLIT SESSION IDENTIFIERS
#    Convert 'OAS30019_UDSb4_d0553' → '3019_0553' → patient_id=3019, session_days=553.
#    This keeps only the numeric parts we need for sorting and merging.
# ─────────────────────────────────────────────────────────────────────────────
# Extract the 4‐digit patient and 4‐digit day directly via regex replace
meta_df['session_label'] = meta_df['OASIS_session_label'].str.replace(
    r'OAS3(\d{4})_.*_d(\d{4})', r'\1_\2', regex=True
)
meta_df[['patient_id','session_days']] = (
    meta_df['session_label']
      .str.split('_', expand=True)
      .astype(int)
)
print("3) Example after cleaning IDs:")
print(meta_df[['OASIS_session_label','session_label','patient_id','session_days']].head(1))

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROPAGATE DIAGNOSES FORWARD IN TIME
#    Enforce “once impaired, always impaired” by taking a cumulative max
#    on the numeric labels within each patient’s time‐sorted visits.
# ─────────────────────────────────────────────────────────────────────────────
meta_df.sort_values(['patient_id','session_days'], inplace=True)
meta_df['propagated_diagnosis_label'] = (
    meta_df
      .groupby('patient_id')['diagnosis_label']
      .cummax()              # never lets the label decrease
      .astype(int)
)
print("4) Label counts after propagation:")
print(meta_df['propagated_diagnosis_label'].value_counts().sort_index())

# ─────────────────────────────────────────────────────────────────────────────
# 5. LOAD YOUR ACTUAL SCAN SESSIONS & ASSIGN LABELS
#    We have a CSV of the scans we actually processed: 'final_sessions.csv'
#    (one row per scan, format pppp_ssss). We split it the same way,
#    sort, then nearest‐neighbor‐merge on session_days to pull in
#    the closest diagnosis for each scan.
# ─────────────────────────────────────────────────────────────────────────────
# 5.1) Load and split
final_df = pd.read_csv('data/connectome_pipeline/final_sessions.csv', header=None, names=['session_label'])
final_df[['patient_id','session_days']] = (
    final_df['session_label']
      .str.split('_', expand=True)
      .astype(int)
)

# 5.2) Sort BOTH tables by (patient_id, session_days)
#     merge_asof requires that the left (final_df) is sorted on the 'on' key
meta_df.sort_values(['patient_id','session_days'], inplace=True)
final_df.sort_values(['patient_id','session_days'], inplace=True)


def assign_labels(df_scans, df_meta):
    # df_scans and df_meta are for _one_ patient, sorted by session_days
    return pd.merge_asof(
        df_scans,
        df_meta[['session_days','propagated_diagnosis_label']],
        on='session_days',
        direction='nearest'
    )

# Prepare both tables, split & cast session_days
# Then:

labeled = (
    final_df
      .groupby('patient_id', group_keys=False)
      .apply(lambda g: assign_labels(g, meta_df[meta_df.patient_id==g.name]))
      .reset_index(drop=True)
)

print(labeled.head())
print(labeled['propagated_diagnosis_label'].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# Now `labeled_scans` contains:
#   - session_label, patient_id, session_days
#   - propagated_diagnosis_label (0/1/2) ready for training.
# ─────────────────────────────────────────────────────────────────────────────

# Save the session labels and their assigned diagnoses to CSV
output_path = 'data/labels.csv'
labeled[['session_label', 'propagated_diagnosis_label']] \
    .to_csv(output_path, index=False)
print(f"Labels written to {output_path}")