"""
Filename: constants.py

Author: Hakima Laribi

Description: Predictor column names and sample identifiers, grouped by prediction task:
    ICU: mortality within 48h in ICU, and length of stay < 48h in ICU (same cohort/predictors)
    30DR: 30-day risk of readmission
Identifiers (SUBJECT/ADM/STAY) and DEMO are shared across tasks.
"""
import pandas as pd

# ICD-10 hierarchy (specific -> general); used only to read off category names for DX_30DR
icd10_cm = pd.read_csv('csvs/utils/30dr_utils/icd10_cm_hierarchy.csv')

# ICU: continuous lab values
LAB_ICU = ['Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine', 'Urea Nitrogen', 'Bicarbonate', 'Anion Gap',
            'Hemoglobin', 'Hematocrit', 'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
            'Calcium, Total',
            'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW', 'Neutrophils', 'Vancomycin']

# ICU: continuous vital-sign/chart values
CHART_ICU = ['Heart Rate', 'NIBP Systolic', 'NIBP Diastolic',
              'NIBP Mean', 'Respiratory Rate', 'O2 Saturation', 'GCS - Verbal',
              'GCS - Eye', 'GCS - Motor']

# ICU: procedure names (discrete, count of occurrences)
PROCEDURE_ICU = ['Foley Catheter', 'PICC Line', 'Intubation', 'Peritoneal Dialysis', 'Bronchoscopy', 'EEG',
                  'Dialysis - CRRT', 'Dialysis Catheter', 'Chest Tube Removed', 'Hemodialysis']

# 30DR: continuous lab values
LAB_30DR = ['pH Other Body Fluid', 'Lymphocytes Joint Fluid', 'Hemoglobin Blood', 'Basophils Joint Fluid',
            'Chloride Blood',
            'Glucose Blood', 'Monocytes Blood', 'Eosinophils Joint Fluid', 'pH Urine', 'Monocytes Ascites',
            'Basophils Cerebrospinal Fluid',
            'Hematocrit Urine', 'Bicarbonate Blood', 'Basophils Other Body Fluid', 'Potassium Blood',
            'Eosinophils Blood', 'Eosinophils Pleural', 'L Blood',
            'Lymphocytes Ascites', 'Creatinine Blood', 'Eosinophils Ascites', 'pO2 Blood',
            'Lymphocytes Other Body Fluid', 'Hematocrit Blood', 'Leukocytes Urine',
            'Calcium, Total Blood', 'Anion Gap Blood', 'Eosinophils Other Body Fluid', 'Sodium Blood',
            'Lymphocytes Blood', 'Basophils Pleural', 'Lactate Blood', 'H Blood',
            'Neutrophils Blood', 'Monocytes Cerebrospinal Fluid', 'Troponin T Blood', 'Platelet Count Blood',
            'Basophils Blood', 'Monocytes Joint Fluid', 'Eosinophils Urine',
            'pH Blood', 'Glucose Urine', 'Eosinophils Cerebrospinal Fluid', 'pCO2 Blood', 'Basophils Ascites',
            'Lymphocytes Pleural']

# 30DR: therapeutic drug classes (discrete, count of occurrences)
MED_30DR = ['ANALGESICS', 'ANESTHETICS', 'ANTIARTHRITICS', 'ANTIASTHMATICS', 'ANTIBIOTICS', 'ANTICOAGULANTS',
            'ANTIDOTES', 'ANTIFUNGALS', 'ANTIHISTAMINES', 'ANTIHYPERGLYCEMICS', 'ANTIINFECTIVES/MISCELLANEOUS',
            'ANTIINFLAM.TUMOR NECROSIS FACTOR INHIBITING AGENTS', 'ANTINEOPLASTICS', 'ANTIPARASITICS',
            'ANTIPARKINSON DRUGS', 'ANTIPLATELET DRUGS', 'ANTIVIRALS', 'AUTONOMIC DRUGS', 'BIOLOGICALS', 'BLOOD',
            'CARDIAC DRUGS', 'CARDIOVASCULAR', 'CNS DRUGS', 'CONTRACEPTIVES', 'COUGH/COLD PREPARATIONS', 'DIAGNOSTIC',
            'DIURETICS', 'EENT PREPS', 'ELECT/CALORIC/H2O', 'GASTROINTESTINAL', 'HORMONES', 'IMMUNOSUPPRESSANTS',
            'MUSCLE RELAXANTS', 'PRE-NATAL VITAMINS', 'PSYCHOTHERAPEUTIC DRUGS', 'SEDATIVE/HYPNOTICS', 'SKIN PREPS',
            'SMOKING DETERRENTS', 'THYROID PREPS', 'UNCLASSIFIED DRUG PRODUCTS', 'VITAMINS']

# 30DR: higher-level procedure categories (discrete, count of occurrences)
PROCEDURE_30DR = ['0 Medical and Surgical - Anatomical Regions, General',
                  '0 Medical and Surgical - Anatomical Regions, Lower Extremities',
                  '0 Medical and Surgical - Anatomical Regions, Upper Extremities',
                  '0 Medical and Surgical - Bursae and Ligaments',
                  '0 Medical and Surgical - Central Nervous System and Cranial Nerves',
                  '0 Medical and Surgical - Ear, Nose, Sinus',
                  '0 Medical and Surgical - Endocrine System',
                  '0 Medical and Surgical - Eye',
                  '0 Medical and Surgical - Female Reproductive System',
                  '0 Medical and Surgical - Gastrointestinal System',
                  '0 Medical and Surgical - Head and Facial Bones',
                  '0 Medical and Surgical - Heart and Great Vessels',
                  '0 Medical and Surgical - Hepatobiliary System and Pancreas',
                  '0 Medical and Surgical - Lower Arteries', '0 Medical and Surgical - Lower Bones',
                  '0 Medical and Surgical - Lower Joints', '0 Medical and Surgical - Lower Veins',
                  '0 Medical and Surgical - Lymphatic and Hemic Systems',
                  '0 Medical and Surgical - Male Reproductive System', '0 Medical and Surgical - Mouth and Throat',
                  '0 Medical and Surgical - Muscles', '0 Medical and Surgical - Peripheral Nervous System',
                  '0 Medical and Surgical - Respiratory System', '0 Medical and Surgical - Skin and Breast',
                  '0 Medical and Surgical - Subcutaneous Tissue and Fascia', '0 Medical and Surgical - Tendons',
                  '0 Medical and Surgical - Upper Arteries', '0 Medical and Surgical - Upper Bones',
                  '0 Medical and Surgical - Upper Joints', '0 Medical and Surgical - Upper Veins',
                  '0 Medical and Surgical - Urinary System', '1 Obstetrics - Pregnancy',
                  '2 Placement - Anatomical Orifices', '2 Placement - Anatomical Regions',
                  '3 Administration - Circulatory', '3 Administration - Indwelling Device',
                  '3 Administration - Physiological Systems and Anatomical Regions',
                  '4 Measurement and Monitoring - Physiological Devices',
                  '4 Measurement and Monitoring - Physiological Systems',
                  '5 Extracorporeal or Systemic Assistance and Performance - Physiological System',
                  '6 Extracorporeal or Systemic Therapies - Physiological Systems',
                  '8 Other Procedures - Indwelling Device',
                  '8 Other Procedures - Physiological Systems and Anatomical Regions', 'B Imaging - Anatomical Regions',
                  'B Imaging - Axial Skeleton, Except Skull and Facial Bones', 'B Imaging - Central Nervous System',
                  'B Imaging - Eye', 'B Imaging - Female Reproductive System', 'B Imaging - Gastrointestinal System',
                  'B Imaging - Heart', 'B Imaging - Hepatobiliary System and Pancreas', 'B Imaging - Lower Arteries',
                  'B Imaging - Lymphatic System', 'B Imaging - Male Reproductive System',
                  'B Imaging - Non-Axial Lower Bones', 'B Imaging - Non-Axial Upper Bones',
                  'B Imaging - Respiratory System', 'B Imaging - Skin, Subcutaneous Tissue and Breast',
                  'B Imaging - Upper Arteries', 'B Imaging - Urinary System', 'B Imaging - Veins',
                  'C Nuclear Medicine - Anatomical Regions', 'C Nuclear Medicine - Central Nervous System',
                  'C Nuclear Medicine - Endocrine System', 'C Nuclear Medicine - Hepatobiliary System and Pancreas',
                  'C Nuclear Medicine - Male Reproductive System',
                  'C Nuclear Medicine - Skin, Subcutaneous Tissue and Breast', 'C Nuclear Medicine - Urinary System',
                  'D Radiation Therapy - Anatomical Regions', 'D Radiation Therapy - Breast',
                  'D Radiation Therapy - Central and Peripheral Nervous System',
                  'D Radiation Therapy - Ear, Nose, Mouth and Throat', 'D Radiation Therapy - Endocrine System',
                  'D Radiation Therapy - Eye', 'D Radiation Therapy - Female Reproductive System',
                  'D Radiation Therapy - Gastrointestinal System',
                  'D Radiation Therapy - Hepatobiliary System and Pancreas',
                  'D Radiation Therapy - Lymphatic and Hematologic System',
                  'D Radiation Therapy - Male Reproductive System', 'D Radiation Therapy - Musculoskeletal System',
                  'D Radiation Therapy - Respiratory System', 'D Radiation Therapy - Skin',
                  'D Radiation Therapy - Urinary System',
                  'F Physical Rehabilitation and Diagnostic Audiology - Rehabilitation',
                  'X New Technology - Anatomical Regions', 'X New Technology - Bones',
                  'X New Technology - Cardiovascular System', 'X New Technology - Extracorporeal',
                  'X New Technology - Muscles, Tendons, Bursae and Ligaments',
                  'X New Technology - Skin, Subcutaneous Tissue, Fascia and Breast']

# 30DR: higher-level diagnosis categories (mapped icd9 -> icd10 -> icd10_cm hierarchy above),
# excluding chapter Z (factors influencing health status, not a diagnosis); binary columns
DX_30DR = icd10_cm[icd10_cm['chapter_idx'] != 'Z']['category'].unique().tolist()

# Shared across tasks
DEMO = ['age', 'gender', 'insurance', 'language', 'marital_status', 'race']

# ICU: image/note embedding sizes and column names
N_IMG_FEAT_ICU = 1042
N_NOTE_FEAT = 768
IMG_COL_ICU = [f'img_last_feat_{i}' for i in range(N_IMG_FEAT_ICU)]

RAD_COL = [f'rad_last_feat_{i}' for i in range(N_NOTE_FEAT)] + [f'rad_agg_feat_{i}' for i in range(N_NOTE_FEAT)]
DISCH_COL = [f'disch_agg_feat_{i}' for i in range(N_NOTE_FEAT)]

# ICU: lab/chart columns, each paired with a "_present" flag; procedures summed into one count
TS_LAB_ICU = [f'{col}{suffix}' for col in LAB_ICU for suffix in ['', '_present']]
TS_CHART_ICU = [f'{col}{suffix}' for col in CHART_ICU for suffix in ['', '_present']]
SUM_PROC_ICU = [f'{col}_sum' for col in PROCEDURE_ICU]

# 30DR: all tabular (non-image, non-note) predictor columns
TABULAR_30DR = LAB_30DR + PROCEDURE_30DR + DX_30DR + MED_30DR

# Sample identifiers: stay_id for ICU, hadm_id (visit id) for 30DR
SUBJECT = 'subject_id'
ADM = 'hadm_id'
STAY = 'stay_id'
