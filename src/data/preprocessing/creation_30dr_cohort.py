"""
Filename: creation_30dr_cohort.py

Author: Hakima Laribi

Description: Creation of the MIMIC-IV 30-day readmission (30DR) cohort. Builds, for each admission with a usable
frontal (AP/PA) chest X-ray, the EHR (labs, procedures, medications, diagnoses), image and note embeddings,
the demographics and the 30-day readmission target. Also provides the utilities to build the ICD-10 hierarchy and
ICD-9 to ICD-10 mapping csvs.
"""
import os
import re
from typing import List, Literal

import pandas as pd

from src.data.preprocessing import constants
from src.data.preprocessing.base_mimic_cohort import CohortMIMIC


class ReadmissionCohortMIMIC(CohortMIMIC):
    """30-day readmission cohort of patients alive at discharge with at least one chest X-ray during admission."""

    def __init__(self,
                 cxr_adm_ids: str,
                 cxr_meta_data: str,
                 admissions: str,
                 radiology_notes: str,
                 discharge_notes: str,
                 pcs_hierarchy: str,
                 cm_hierarchy: str,
                 icd9_to_icd10_cm: str,
                 icd9_to_icd10_pcs: str) -> None:
        """
        Args:
            cxr_adm_ids: path to the csv of the cohort admissions with demographics and readmission target
            cxr_meta_data: path to the MIMIC-CXR-JPG metadata csv
            admissions: path to the MIMIC-IV admissions csv
            radiology_notes: path to the MIMIC-IV-Note radiology csv
            discharge_notes: path to the MIMIC-IV-Note discharge csv
            pcs_hierarchy: path to the ICD-10-PCS hierarchy csv (see create_icd10_pcs_hierarchy)
            cm_hierarchy: path to the ICD-10-CM hierarchy csv
            icd9_to_icd10_cm: path to the ICD-9 to ICD-10-CM mapping csv (see from_txt_to_csv)
            icd9_to_icd10_pcs: path to the ICD-9 to ICD-10-PCS mapping csv (see from_txt_to_csv)
        """
        # Set protected attributes
        self._cxr_adm_ids = pd.read_csv(cxr_adm_ids)
        self._cxr_meta_data = pd.read_csv(cxr_meta_data)
        self._admissions = pd.read_csv(admissions)
        self._radiology_notes = pd.read_csv(radiology_notes)
        self._discharge_notes = pd.read_csv(discharge_notes)
        self._pcs_hierarchy = pd.read_csv(pcs_hierarchy, dtype=str)
        self._cm_hierarchy = pd.read_csv(cm_hierarchy, dtype=str)
        self._icd9_to_icd10_cm = pd.read_csv(icd9_to_icd10_cm, dtype=str)
        self._icd9_to_icd10_pcs = pd.read_csv(icd9_to_icd10_pcs, dtype=str)

        # Keep only patients alive at discharge
        self._cxr_adm_ids = self._cxr_adm_ids[(self._cxr_adm_ids['deathtime'].isna())
                                              & (self._cxr_adm_ids['discharge_location'] != 'DIED')]

        # Transform time columns to date time
        self._admissions = self.transform_to_datetime(self._admissions, ['admittime', 'dischtime', 'edregtime'])
        self._cxr_adm_ids = self.transform_to_datetime(self._cxr_adm_ids, ['admittime', 'dischtime', 'edregtime'])
        self._radiology_notes = self.transform_to_datetime(self._radiology_notes, ['charttime'])
        self._discharge_notes = self.transform_to_datetime(self._discharge_notes, ['charttime'])

        # Add the number of days each admission lasted
        self._cxr_adm_ids['length_of_stay'] = ((self._cxr_adm_ids['dischtime'] - self._cxr_adm_ids['edregtime'].
                                                combine_first(self._cxr_adm_ids['admittime']))
                                               .dt.total_seconds() // (24 * 3600)).astype(int) + 1

        # Keep only patients included in the 30DR study
        subjects_ids = self._cxr_adm_ids['subject_id'].unique().tolist()
        self._cxr_meta_data = self._cxr_meta_data[self._cxr_meta_data['subject_id'].isin(subjects_ids)]
        self._radiology_notes = self._radiology_notes[self._radiology_notes['subject_id'].isin(subjects_ids)]
        self._discharge_notes = self._discharge_notes[self._discharge_notes['subject_id'].isin(subjects_ids)]
        self._admissions = self._admissions[self._admissions['hadm_id'].isin(self._cxr_adm_ids['hadm_id'])]

        # Get ids and time of images included in the study
        self._imaging_ids = self.get_study_images()

    def __call__(self,
                 labs: pd.DataFrame,
                 diagnoses: pd.DataFrame,
                 procedures: pd.DataFrame,
                 medications: pd.DataFrame,
                 dense_features_path: List[str],
                 rad_features_path: List[str],
                 disch_features_path: List[str],
                 saving_file: str,
                 empty_string_embed: pd.DataFrame):
        """
        Builds the cohort and saves 30dr_cxr.csv (one image per admission and image time) and
        30dr_notes_ehr_demo.csv (EHR, diagnoses, notes, demographics and target) in saving_file.

        Args:
            labs: lab events
            diagnoses: diagnoses ICD codes
            procedures: procedures ICD codes
            medications: medication events
            dense_features_path: paths to the image embedding csvs
            rad_features_path: paths to the radiology note embedding csvs
            disch_features_path: paths to the discharge note embedding csvs
            saving_file: directory where the csvs are saved
            empty_string_embed: one-row dataframe with the embedding of an empty note
        """
        # Get modalities embeddings
        ehr_embeddings, dx_embeddings = self.get_ehr_embeddings(labs, procedures, diagnoses, medications)
        cxr_embeddings = self.get_cxr_embeddings(dense_features_path)
        rad_embeddings = self.get_notes_embeddings(rad_features_path, 'rad')
        disch_embeddings = self.get_notes_embeddings(disch_features_path, 'disch')

        # get admissions where an imaging occurred
        cxr_adm = cxr_embeddings['hadm_id'].unique().tolist()

        # Save imaging modality
        cxr_embeddings = cxr_embeddings[['subject_id', 'hadm_id', 'dicom_id', 'img_time', 'ViewPosition'] +
                                        [f'feature_{i}' for i in range(constants.N_IMG_FEAT_ICU)]]
        cxr_embeddings.to_csv(os.path.join(saving_file, '30dr_cxr.csv'), index=False)

        # Concatenate tabular and text modalities
        patients_embed = pd.concat([ehr_embeddings, dx_embeddings, rad_embeddings, disch_embeddings],
                                   axis=1).reset_index()

        # Keep only patients who had an imaging
        patients_embed = patients_embed[patients_embed['hadm_id'].isin(cxr_adm)]

        # Impute missing notes
        patients_embed = self.impute_notes(patients_embed, empty_string_embed,
                                           [f'rad_{i}' for i in range(constants.N_NOTE_FEAT)])
        patients_embed = self.impute_notes(patients_embed, empty_string_embed,
                                           [f'disch_{i}' for i in range(constants.N_NOTE_FEAT)])

        # Add demographics and target
        patients_embed = self.get_demographics_targets(patients_embed)
        patients_embed.to_csv(os.path.join(saving_file, '30dr_notes_ehr_demo.csv'), index=False)

    def get_demographics_targets(self,
                                 patients_embed: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            patients_embed: features with subject_id and hadm_id columns

        Returns:
            the features with demographics, length_of_stay and the '30dr' target
        """
        demo_target = ['subject_id', 'hadm_id'] + constants.DEMO + ['length_of_stay', 'readmitted_within_30days']
        patients_embed = self._cxr_adm_ids[demo_target].merge(patients_embed, on=['subject_id', 'hadm_id'], how='inner')
        return patients_embed.rename(columns={'readmitted_within_30days': '30dr'})

    def get_ehr_embeddings(self,
                           labs: pd.DataFrame,
                           procedures: pd.DataFrame,
                           diagnoses: pd.DataFrame,
                           medications: pd.DataFrame):
        """
        Args:
            labs: lab events
            procedures: procedures ICD codes
            diagnoses: diagnoses ICD codes
            medications: medication events

        Returns:
            ehr_embed: labs, procedures and medications of the cohort admissions
            dx_embed: diagnoses of the cohort admissions
        """
        index_cols = ['subject_id', 'hadm_id']
        # Transform tabular data types
        lab_embeddings = self.create_lab_cols(labs).set_index(index_cols)
        proc_embeddings = self.create_code_cols(procedures, 'procedures').set_index(index_cols)
        diag_embeddings = self.create_code_cols(diagnoses, 'diagnoses')
        med_embeddings = self.create_medication_cols(medications).set_index(index_cols)

        # concatenate EHR features
        ehr_embed = pd.concat([lab_embeddings, proc_embeddings, med_embeddings], axis=1).reset_index()

        # Merge with the admission ids of the study
        ehr_embed = ehr_embed.merge(self._cxr_adm_ids[['subject_id', 'hadm_id']],
                                    how='right',
                                    on=['subject_id', 'hadm_id'])

        # Process diagnoses individually
        dx_embed = diag_embeddings.merge(self._cxr_adm_ids[['subject_id', 'hadm_id']],
                                         how='right',
                                         on=['subject_id', 'hadm_id'])
        dx_embed[diag_embeddings.columns] = dx_embed[diag_embeddings.columns].fillna('0')  # fill missing values

        # fill missing values: only the presence indicator of labs, procedures and medications
        labs_cols = [f for f in lab_embeddings.columns if 'present' in f]
        ehr_embed = self.impute_events(0, ehr_embed, labs_cols)
        ehr_embed = self.impute_events(0, ehr_embed, proc_embeddings.columns)
        ehr_embed = self.impute_events(0, ehr_embed, med_embeddings.columns)

        # Set indexes
        ehr_embed = ehr_embed.set_index(index_cols)
        dx_embed = dx_embed.set_index(index_cols)

        return ehr_embed, dx_embed

    @staticmethod
    def impute_events(fill_with: object,
                      ehr_embed: pd.DataFrame,
                      lab_columns: List[str]) -> pd.DataFrame:
        """
        Args:
            fill_with: value used to fill missing values
            ehr_embed: events dataframe
            lab_columns: columns to fill

        Returns:
            the events with filled columns
        """
        # fill missing values
        ehr_embed[lab_columns] = ehr_embed[lab_columns].fillna(fill_with)
        return ehr_embed

    def create_lab_cols(self, labevents: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots lab events to one column per lab (last value), with a {lab}_present indicator.

        Args:
            labevents: lab events with label_fluid, charttime and last_value columns

        Returns:
            pivoted lab events
        """
        print('Extracting lab event ..')
        index_cols = ['subject_id', 'hadm_id']

        labevents['charttime'] = pd.to_datetime(labevents['charttime'])
        # Sort values
        labevents = labevents.sort_values(by='charttime', ascending=True).reset_index(drop=True)

        # create new dataframes of lab events
        df_pivoted = self.pivot_dataset(labevents,
                                        index_cols,
                                        'label_fluid',
                                        'last_value')

        for col in labevents['label_fluid'].unique().tolist():
            # Create missing indicator
            df_pivoted[f'{col}_present'] = df_pivoted[col].notna().astype(int)
        return df_pivoted

    @staticmethod
    def pivot_dataset(df: pd.DataFrame,
                      index_cols: List[str],
                      code_cols: str,
                      value_col: str,
                      agg_type: Literal['sum', 'last'] = 'last') -> pd.DataFrame:
        """
        Args:
            df: events dataframe
            index_cols: rows of the pivoted dataframe
            code_cols: column whose values become the columns
            value_col: column of the values
            agg_type: aggregation of multiple values

        Returns:
            the pivoted dataframe
        """
        df_pivoted = df.pivot_table(
            index=index_cols,
            columns=code_cols,
            values=value_col,
            aggfunc=agg_type
        ).reset_index()

        df_pivoted.columns.name = None
        return df_pivoted

    def create_code_cols(self,
                         df_codes: pd.DataFrame,
                         data_type: Literal['procedures', 'diagnoses']) -> pd.DataFrame:
        """
        Maps ICD codes to ICD-10 categories and pivots them to one column per category.

        Args:
            df_codes: ICD codes with icd_code and icd_version columns
            data_type: 'procedures' or 'diagnoses'

        Returns:
            pivoted codes (binary for diagnoses, counts for procedures)
        """
        print(f'Extracting {data_type} codes......')
        # First, convert icd9 to their icd10 mapping according to General Equivalence Mapping (GEM)
        df_mapped = self.map_icd9_to_icd10(df_codes, data_type)
        # Map icd10 codes to parent subgroups to reduce granularity
        df_hrc = self.map_icd10_to_parent(df_mapped, data_type)
        return df_hrc

    def map_icd9_to_icd10(self, df_code: pd.DataFrame,
                          data_type: Literal['procedures', 'diagnoses']) -> pd.DataFrame:
        """
        Converts ICD-9 codes to ICD-10 with the General Equivalence Mapping (GEM).
        Unmapped codes are retried with a trailing '0' (e.g. V127 -> V1270).

        Args:
            df_code: ICD codes with icd_code and icd_version columns
            data_type: 'procedures' or 'diagnoses'

        Returns:
            subject_id, hadm_id, icd10 (and chartdate for procedures)
        """
        # Get the conversion table of codes
        convert_table = self._icd9_to_icd10_cm if data_type == 'diagnoses' else self._icd9_to_icd10_pcs
        # Remove spaces
        df_code['icd_code'] = df_code['icd_code'].str.replace(' ', '', regex=True)
        convert_table['icd9'] = convert_table['icd9'].str.replace(' ', '')
        convert_table['icd10'] = convert_table['icd10'].str.replace(' ', '')
        convert_table['flag'] = convert_table['flag'].str.replace(' ', '')

        # Filter ICD-9 codes
        icd9_df = df_code[df_code['icd_version'] == 9].copy()

        # Merge with conversion table
        merged = icd9_df.merge(convert_table, how='left', left_on='icd_code', right_on='icd9')

        # Warn about unmapped ICD-9 codes
        missing = merged[merged['icd10'].isna()][df_code.columns]
        # Sometimes the code is too general here (V127) make it more specific for mapping
        if missing.shape[0] > 0:
            missing['icd_code'] = missing['icd_code'].astype(str) + '0'
            merge_miss = missing.merge(convert_table, how='left', left_on='icd_code', right_on='icd9')
            merged.loc[missing.index] = merge_miss.values
            missing = merged[merged['icd10'].isna()]
            if missing.shape[0] > 0:
                print(f'WARNING: {len(missing)} ICD-9 codes not found in ICD-10 table:')
                print(", ".join(missing['icd_code'].astype(str)))

        # Use the mapped icd10 code where available
        df_codes = pd.concat([df_code[df_code['icd_version'] == 10].rename(columns={'icd_code': 'icd10'}), merged])
        columns = ['subject_id', 'hadm_id', 'icd10']
        if data_type == 'procedures':
            columns.append('chartdate')
        return df_codes[columns]

    def map_icd10_to_parent(self,
                            df_icd10: pd.DataFrame,
                            data_type: Literal['procedures', 'diagnoses']) -> pd.DataFrame:
        """
        Maps ICD-10 codes to their parent category and pivots them.

        Args:
            df_icd10: codes with an icd10 column
            data_type: 'procedures' or 'diagnoses'

        Returns:
            pivoted categories (binary for diagnoses, counts for procedures)
        """
        # Get function which maps codes to parents in ontology
        hierarchize = self.hierarchize_cm if data_type == 'diagnoses' else self.hierarchize_pcs
        df_hrc = hierarchize(df_icd10)

        # create one column per category
        df_hrc = df_hrc.assign(value=1)
        agg_type = 'sum' if data_type == 'procedures' else 'last'
        df_hrc = self.pivot_dataset(df_hrc, ['subject_id', 'hadm_id'], 'category', 'value', agg_type=agg_type)

        return df_hrc

    def hierarchize_cm(self, df_icd10: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df_icd10: diagnoses with an icd10 column

        Returns:
            the diagnoses (chapter Z excluded) joined with their ICD-10-CM category
        """
        # Add columns to facilitate join
        df_icd10['start_idx'] = df_icd10['icd10'].str[0]
        # exclude codes from chapter Z: Factors influencing health status and contact with health services
        df_icd10 = df_icd10[df_icd10['start_idx'] != 'Z']
        x = df_icd10.merge(self._cm_hierarchy, left_on='start_idx', right_on='chapter_idx').drop('start_idx', axis=1)
        # Find the intervals to which belongs each code,
        x = x[(x['icd10'].str[0] == x['chapter_idx'])
              & (x['icd10'].str[:3] <= x['category_idx'].str[4:])
              & (x['icd10'].str[:3] >= x['category_idx'].str[:3])]
        return x

    def hierarchize_pcs(self,
                        df_icd10: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df_icd10: procedures with an icd10 column

        Returns:
            the procedures joined with their ICD-10-PCS category (first 2 characters)
        """
        # Add columns to facilitate join
        df_icd10['start_idx'] = df_icd10['icd10'].str[:2]
        x = df_icd10.merge(self._pcs_hierarchy, left_on='start_idx', right_on='category_idx').drop('start_idx', axis=1)
        return x

    def create_medication_cols(self, medications: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots medications to one count column per therapeutic class.

        Args:
            medications: medication events with starttime and med_therapeutic_class_description columns

        Returns:
            pivoted medication counts
        """
        print('Extracting medications ..')
        # sort values by time
        medications = medications.sort_values(by='starttime', ascending=True).reset_index(drop=True)

        # create a pivoted dataframe of events
        medications = medications.assign(value=1)

        df_pivoted = self.pivot_dataset(medications,
                                        ['subject_id', 'hadm_id'],
                                        'med_therapeutic_class_description',
                                        'value',
                                        agg_type='sum')

        return df_pivoted

    def get_cxr_embeddings(self, dense_features_path: List[str]) -> pd.DataFrame:
        """
        Args:
            dense_features_path: paths to the image embedding csvs

        Returns:
            embeddings of one image per admission and image time (AP preferred over PA), with admission ids,
            img_time, ViewPosition and study_id
        """
        print('Getting CXR embeddings...')
        # Get the dense features of the images included in our study
        df_img = self.read_dense_features(dense_features_path).drop(columns='pt_group')

        # Ungroup dicom_ids
        exploded_img_ids = self._imaging_ids.explode('dicom_id').reset_index(drop=True)

        # Merge based on dicom_id to get admission ids
        id_images = ['subject_id', 'hadm_id', 'StudyDateTime', 'dicom_id']
        df_img = exploded_img_ids[id_images].merge(df_img, on=['subject_id', 'dicom_id'], how='inner')
        df_img = df_img.rename(columns={'StudyDateTime': 'img_time'})

        # Add position view
        df_img = self._cxr_meta_data[['dicom_id', 'ViewPosition', 'study_id']].merge(df_img, how='inner')

        # Sort images by time
        df_img = df_img.sort_values(by=['subject_id', 'hadm_id', 'img_time', 'study_id'],
                                    ascending=True).reset_index(drop=True)

        return self.sample_one_single_view(df_img, ['subject_id', 'hadm_id'])

    def get_notes_embeddings(self,
                             note_features_path: List[str],
                             note_type: Literal['rad', 'disch']) -> pd.DataFrame:
        """
        Args:
            note_features_path: paths to the note embedding csvs
            note_type: 'rad' (radiology notes, averaged per admission) or 'disch' (discharge, one note per admission)

        Returns:
            note embeddings indexed by (subject_id, hadm_id), with columns renamed to {note_type}_{i}
        """
        # read note embeddings
        notes = 'radiology' if note_type == 'rad' else 'discharge'
        print(f'Getting {notes} notes embeddings...')
        df_note_embeddings = self.read_note_features(note_features_path, notes)

        note_features = [f'feature_{i}' for i in range(constants.N_NOTE_FEAT)]

        # sort values by time
        df_note_embeddings = df_note_embeddings.sort_values(by='charttime', ascending=True).reset_index(drop=True)

        if note_type == 'rad':  # average the radiology notes of each admission
            grouped_embed = df_note_embeddings.groupby(['subject_id', 'hadm_id'])[note_features].mean()
        else:  # for discharge notes, there is only one note per admission, so just get it without aggregation
            grouped_embed = df_note_embeddings[['subject_id', 'hadm_id'] + note_features].set_index(
                ['subject_id', 'hadm_id'])

        return grouped_embed.rename(columns={f'feature_{i}': f'{note_type}_{i}' for i in range(len(note_features))})


def create_icd10_pcs_hierarchy(saving_dir: str = 'csvs/utils/30dr_utils'):
    """
    Creates a csv with the second level of generality of icd10 procedure codes

    Args:
        saving_dir: directory where icd10_pcs_hierarchy.csv is saved
    """
    children = [
        "00 Central Nervous System and Cranial Nerves",
        "01 Peripheral Nervous System",
        "02 Heart and Great Vessels",
        "03 Upper Arteries",
        "04 Lower Arteries",
        "05 Upper Veins",
        "06 Lower Veins",
        "07 Lymphatic and Hemic Systems",
        "08 Eye",
        "09 Ear, Nose, Sinus",
        "0B Respiratory System",
        "0C Mouth and Throat",
        "0D Gastrointestinal System",
        "0F Hepatobiliary System and Pancreas",
        "0G Endocrine System",
        "0H Skin and Breast",
        "0J Subcutaneous Tissue and Fascia",
        "0K Muscles",
        "0L Tendons",
        "0M Bursae and Ligaments",
        "0N Head and Facial Bones",
        "0P Upper Bones",
        "0Q Lower Bones",
        "0R Upper Joints",
        "0S Lower Joints",
        "0T Urinary System",
        "0U Female Reproductive System",
        "0V Male Reproductive System",
        "0W Anatomical Regions, General",
        "0X Anatomical Regions, Upper Extremities",
        "0Y Anatomical Regions, Lower Extremities",
        "10 Pregnancy",
        "2W Anatomical Regions",
        "2Y Anatomical Orifices",
        "30 Circulatory",
        "3C Indwelling Device",
        "3E Physiological Systems and Anatomical Regions",
        "4A Physiological Systems",
        "4B Physiological Devices",
        "5A Physiological System",
        "6A Physiological Systems",
        "7W Anatomical Regions",
        "8C Indwelling Device",
        "8E Physiological Systems and Anatomical Regions",
        "9W Anatomical Regions",
        "B0 Central Nervous System",
        "B2 Heart",
        "B3 Upper Arteries",
        "B4 Lower Arteries",
        "B5 Veins",
        "B7 Lymphatic System",
        "B8 Eye",
        "B9 Ear, Nose, Mouth and Throat",
        "BB Respiratory System",
        "BD Gastrointestinal System",
        "BF Hepatobiliary System and Pancreas",
        "BG Endocrine System",
        "BH Skin, Subcutaneous Tissue and Breast",
        "BL Connective Tissue",
        "BN Skull and Facial Bones",
        "BP Non-Axial Upper Bones",
        "BQ Non-Axial Lower Bones",
        "BR Axial Skeleton, Except Skull and Facial Bones",
        "BT Urinary System",
        "BU Female Reproductive System",
        "BV Male Reproductive System",
        "BW Anatomical Regions",
        "BY Fetus and Obstetrical",
        "C0 Central Nervous System",
        "C2 Heart",
        "C5 Veins",
        "C7 Lymphatic and Hematologic System",
        "C8 Eye",
        "C9 Ear, Nose, Mouth and Throat",
        "CB Respiratory System",
        "CD Gastrointestinal System",
        "CF Hepatobiliary System and Pancreas",
        "CG Endocrine System",
        "CH Skin, Subcutaneous Tissue and Breast",
        "CP Musculoskeletal System",
        "CT Urinary System",
        "CV Male Reproductive System",
        "CW Anatomical Regions",
        "D0 Central and Peripheral Nervous System",
        "D7 Lymphatic and Hematologic System",
        "D8 Eye",
        "D9 Ear, Nose, Mouth and Throat",
        "DB Respiratory System",
        "DD Gastrointestinal System",
        "DF Hepatobiliary System and Pancreas",
        "DG Endocrine System",
        "DH Skin",
        "DM Breast",
        "DP Musculoskeletal System",
        "DT Urinary System",
        "DU Female Reproductive System",
        "DV Male Reproductive System",
        "DW Anatomical Regions",
        "F0 Rehabilitation",
        "F1 Diagnostic Audiology",
        "X2 Cardiovascular System",
        "XH Skin, Subcutaneous Tissue, Fascia and Breast",
        "XK Muscles, Tendons, Bursae and Ligaments",
        "XN Bones",
        "XR Joints",
        "XT Urinary System",
        "XV Male Reproductive System",
        "XW Anatomical Regions",
        "XX Physiological Systems",
        "XY Extracorporeal"
    ]
    parents = [
        "0 Medical and Surgical",
        "1 Obstetrics",
        "2 Placement",
        "3 Administration",
        "4 Measurement and Monitoring",
        "5 Extracorporeal or Systemic Assistance and Performance",
        "6 Extracorporeal or Systemic Therapies",
        "7 Osteopathic",
        "8 Other Procedures",
        "9 Chiropractic",
        "B Imaging",
        "C Nuclear Medicine",
        "D Radiation Therapy",
        "F Physical Rehabilitation and Diagnostic Audiology",
        "G Mental Health",
        "H Substance Abuse Treatment",
        "X New Technology"
    ]
    df = pd.DataFrame(columns=['chapter_idx', 'chapter', 'category_idx', 'category'])
    for group in parents:
        data = {}
        data['category_idx'] = [subgroup[:2] for subgroup in children if subgroup[0] == group[0]]
        data['category'] = [f'{group} - {subgroup[3:]}' for subgroup in children if subgroup[0] == group[0]]
        data['chapter_idx'] = [group[0]] * len(data['category_idx'])
        data['chapter'] = [group[2:]] * len(data['category_idx'])
        df = pd.concat([df, pd.DataFrame.from_dict(data)], ignore_index=True)

    df.to_csv(os.path.join(saving_dir, 'icd10_pcs_hierarchy.csv'), index=False)


def from_txt_to_csv(input_file: str):
    """
    Converts txt files of General Equivalence Mapping of ICD9 codes to ICD10 codes to csv files for easier manipulations

    Args:
        input_file: path to the GEM .txt file; the csv is saved next to it
    """
    # Regular expression to parse lines like "0010  A000    00000"
    pattern = re.compile(r'(\S+)\s+(\S+)\s+(\S+)')

    # Read and parse lines
    with open(input_file, 'r') as f:
        rows = [pattern.match(line.strip()).groups() for line in f if pattern.match(line.strip())]

    # Create DataFrame
    df = pd.DataFrame(rows, columns=['icd9', 'icd10', 'flag']).astype(str)

    # Save to CSV
    df.to_csv(input_file[:-3] + 'csv', index=False)
