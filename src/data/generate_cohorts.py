"""
Filename: generate_cohorts.py

Author: Hakima Laribi

Description: Generates the 30-day readmission (30DR) and ICU cohorts from the MIMIC-IV, MIMIC-CXR-JPG and
MIMIC-IV-Note data, the image embeddings (src/data/extraction/extraction_mimic_cxr.py) and the note embeddings
(src/data/extraction/extraction_mimic_notes.py).

Run from the root of the repository:
    python -m src.data.generate_cohorts

Paths below are relative to the root of the repository; edit them to match where your data is stored.
"""
import pandas as pd

from src.data.preprocessing.creation_30dr_cohort import ReadmissionCohortMIMIC
from src.data.preprocessing.creation_icu_cohort import ICUCohortMIMIC

# Directory where the cohort csvs are saved
OUTPUT_DIR = 'csvs/datasets'

# MIMIC-IV, MIMIC-CXR-JPG and MIMIC-IV-Note tables
CXR_META_DATA = 'physionet.org/image_features/mimic-cxr-2.0.0-metadata.csv'
ADMISSIONS = 'physionet.org/mimic-iv-3.1/hosp/admissions.csv'
ICUSTAYS = 'physionet.org/mimic-iv-3.1/icu/icustays.csv'
RADIOLOGY_NOTES = 'physionet.org/note/radiology.csv'
DISCHARGE_NOTES = 'physionet.org/note/discharge.csv'

# Image and note embeddings
IMG_FEATURES = [f'physionet.org/image_features/cxr_features/cxr_pt_group_p{i}.csv' for i in range(10, 20)]
RAD_FEATURES = [f'physionet.org/note/radiology/Bio_ClinicalBERT_features_radiology_{i}.csv' for i in range(25)]
DISCH_FEATURES = [f'physionet.org/note/discharge/Bio_ClinicalBERT_features_discharge_{i}.csv' for i in range(5)]
EMPTY_NOTE_FEATURES = 'physionet.org/note/radiology/Bio_ClinicalBERT_features_empty.csv'

# Task-specific inputs
DIR_30DR = 'csvs/utils/30dr_modalities'
DIR_30DR_UTILS = 'csvs/utils/30dr_utils'
DIR_ICU = 'csvs/utils/icu_modalities'


def generate_30dr_cohort():
    """Generates 30dr_cxr.csv and 30dr_notes_ehr_demo.csv in OUTPUT_DIR."""
    print('extraction of 30DR cohort ...')
    cohort = ReadmissionCohortMIMIC(cxr_adm_ids=f'{DIR_30DR}/demo_target.csv',
                                    cxr_meta_data=CXR_META_DATA,
                                    admissions=ADMISSIONS,
                                    radiology_notes=RADIOLOGY_NOTES,
                                    discharge_notes=DISCHARGE_NOTES,
                                    icd9_to_icd10_cm=f'{DIR_30DR_UTILS}/2018_I9gem_cm.csv',
                                    icd9_to_icd10_pcs=f'{DIR_30DR_UTILS}/2018_I9gem_pcs.csv',
                                    cm_hierarchy=f'{DIR_30DR_UTILS}/icd10_cm_hierarchy.csv',
                                    pcs_hierarchy=f'{DIR_30DR_UTILS}/icd10_pcs_hierarchy.csv')

    cohort(labs=pd.read_csv(f'{DIR_30DR}/labs_time.csv'),
           diagnoses=pd.read_csv(f'{DIR_30DR}/diagnoses.csv'),
           procedures=pd.read_csv(f'{DIR_30DR}/procedures.csv'),
           medications=pd.read_csv(f'{DIR_30DR}/medication.csv'),
           dense_features_path=IMG_FEATURES,
           rad_features_path=RAD_FEATURES,
           disch_features_path=DISCH_FEATURES,
           saving_file=OUTPUT_DIR,
           empty_string_embed=pd.read_csv(EMPTY_NOTE_FEATURES))


def generate_icu_cohort():
    """Generates icu_cxr_24h_mode.csv and icu_rad_demo_events_24h_mode.csv in OUTPUT_DIR."""
    print('extraction of ICU cohort ...')
    cohort = ICUCohortMIMIC(cxr_adm_ids=f'{DIR_ICU}/cxr_adm_ids.csv',
                            cxr_meta_data=CXR_META_DATA,
                            admissions=ADMISSIONS,
                            icustays=ICUSTAYS,
                            radiology_notes=RADIOLOGY_NOTES)

    cohort(demographics=pd.read_csv(f'{DIR_ICU}/demographics.csv'),
           lab=pd.read_csv(f'{DIR_ICU}/labevents.csv'),
           chart=pd.read_csv(f'{DIR_ICU}/chartevents.csv'),
           procedure=pd.read_csv(f'{DIR_ICU}/procedureevents.csv'),
           dense_features_path=IMG_FEATURES,
           note_features_path=RAD_FEATURES,
           saving_file=OUTPUT_DIR,
           empty_string_embed=pd.read_csv(EMPTY_NOTE_FEATURES))


if __name__ == '__main__':
    generate_30dr_cohort()
    generate_icu_cohort()
