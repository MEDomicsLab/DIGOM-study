"""
Filename: base_mimic_cohort.py

Author: Hakima Laribi

Description: Abstract base class of the MIMIC-IV cohorts. Loads the shared MIMIC tables and provides the helpers
used by the cohort builders (for each task) to select study images, read image and note embeddings and impute notes.
"""
from abc import ABC, abstractmethod
from typing import List, Literal

import pandas as pd


class CohortMIMIC(ABC):
    """Base class of the MIMIC-IV multimodal cohorts (CXR images, notes and admissions)."""

    @abstractmethod
    def __init__(self,
                 cxr_adm_ids: str,
                 cxr_meta_data: str,
                 admissions: str,
                 radiology_notes: str,
                 discharge_notes: str):
        """
        Args:
            cxr_adm_ids: path to the csv of the cohort admissions (subject_id, hadm_id, ...)
            cxr_meta_data: path to the MIMIC-CXR-JPG metadata csv
            admissions: path to the MIMIC-IV admissions csv
            radiology_notes: path to the MIMIC-IV-Note radiology csv
            discharge_notes: path to the MIMIC-IV-Note discharge csv
        """
        # Set protected attributes
        self._cxr_adm_ids = pd.read_csv(cxr_adm_ids)
        self._cxr_meta_data = pd.read_csv(cxr_meta_data)
        self._admissions = pd.read_csv(admissions)
        self._radiology_notes = pd.read_csv(radiology_notes)
        self._discharge_notes = pd.read_csv(discharge_notes)

        # Get ids and time of images included in the study
        self._imaging_ids = self.get_study_images()

    def get_study_images(self) -> pd.DataFrame:
        """
        Selects the frontal (AP and PA) CXR images taken during the cohort admissions.

        Returns:
            one row per (subject_id, hadm_id, StudyDateTime) with the tuples of study_id and dicom_id taken at that time
        """
        # Keep only AP and PA images
        df = self._cxr_meta_data[(self._cxr_meta_data['ViewPosition'] == 'AP') |
                                 (self._cxr_meta_data['ViewPosition'] == 'PA')]

        # Get imaging information of multimodal patients
        df = df.merge(self._admissions, on='subject_id', how='inner')
        df = df[df['hadm_id'].isin(self._cxr_adm_ids.hadm_id)]
        df = df.reset_index(drop=True)

        # Ensure datetime columns are in proper format
        df = self.create_cxr_datetime_column(df)
        df['admittime'] = pd.to_datetime(df['admittime'])
        df['edregtime'] = pd.to_datetime(df['edregtime'])
        df['dischtime'] = pd.to_datetime(df['dischtime'])

        # Get the images included in the study (images that happened during a patient admission that had an ICU stay)
        df_img_adm = df[((df['StudyDateTime'] >= df['admittime']) | (df['StudyDateTime'] >= df['edregtime']))
                        & (df['StudyDateTime'] <= df['dischtime'])]

        # Group images that were taken at the same time
        id_columns = ['subject_id', 'hadm_id', 'StudyDateTime']
        img_columns = ['study_id', 'dicom_id']
        df_img_grouped = df_img_adm.groupby(id_columns)[img_columns].agg({col: lambda x: tuple(x.unique().tolist())
                                                                          for col in img_columns})

        df_img_grouped = df_img_grouped.reset_index()
        return df_img_grouped

    def read_dense_features(self,
                            dense_features_path: List[str]) -> pd.DataFrame:
        """
        Reads the image embeddings and keeps those of the study images.

        Args:
            dense_features_path: paths to the per-group image embedding csvs

        Returns:
            image embeddings with subject_id, study_id and dicom_id columns
        """
        df_img = pd.DataFrame()
        # Get study dicom_ids
        study_dicom = [dicom_id for tuple_ids in self._imaging_ids['dicom_id'].values for dicom_id in tuple_ids]
        # read each group of images
        for img_grp in dense_features_path:
            df_grp_img = pd.read_csv(img_grp)
            df_grp_img['n_patient'] = df_grp_img['n_patient'].astype(str).str[1:].astype(int)
            df_grp_img['n_study'] = df_grp_img['n_study'].astype(str).str[1:].astype(int)
            # Keep only dicom ids that were selected for the study
            df_grp_img['image_name'] = df_grp_img['image_name'].astype(str).str[:-4]
            df_grp_img = df_grp_img[df_grp_img['image_name'].isin(study_dicom)]
            df_img = pd.concat([df_img, df_grp_img], ignore_index=True, axis=0)

        df_img = df_img.rename(columns={'n_patient': 'subject_id',
                                        'n_study': 'study_id',
                                        'image_name': 'dicom_id'})
        return df_img

    def read_note_features(self,
                           note_features_path: List[str],
                           note_type: Literal['radiology', 'discharge']) -> pd.DataFrame:
        """
        Reads the note embeddings of the cohort patients and links them to their admission.

        Args:
            note_features_path: paths to the note embedding csvs
            note_type: 'radiology' (kept if written during the admission) or 'discharge' (linked by hadm_id)

        Returns:
            note embeddings with their charttime and admission information
        """
        df_note = pd.DataFrame()
        for note_grp in note_features_path:
            chunk_note = pd.read_csv(note_grp)
            # Keep cohort patients only and remove hadm_ids since it is null sometimes
            chunk_note = chunk_note[chunk_note['subject_id'].isin(self._cxr_adm_ids['subject_id'])].drop('hadm_id',
                                                                                                         axis=1)
            df_note = pd.concat([df_note, chunk_note], ignore_index=True, axis=0)

        # Merge with admissions to get admission ids
        adm_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'edregtime']

        if note_type == 'radiology':
            # Merge with radiology notes csv to get charttime
            df_note = pd.merge(self._radiology_notes[['note_id', 'charttime']], df_note, how='inner', on='note_id')
            df_note_adm = pd.merge(self._admissions[adm_cols], df_note, how='inner', on='subject_id')

            # Keep only notes that happened during that admission
            df_note_adm = df_note_adm[((df_note_adm['charttime'] >= df_note_adm['admittime']) |
                                       (df_note_adm['charttime'] >= df_note_adm['edregtime']))
                                      & (df_note_adm['charttime'] <= df_note_adm['dischtime'])]
        else:  # discharge
            # Merge with discharge notes csv to get charttime and admission id
            df_note = pd.merge(self._discharge_notes[['note_id', 'hadm_id', 'charttime']], df_note, how='inner',
                               on='note_id')
            df_note_adm = pd.merge(self._admissions[adm_cols], df_note, how='inner', on=['subject_id', 'hadm_id'])

        return df_note_adm

    @staticmethod
    def transform_to_datetime(df: pd.DataFrame,
                              columns: List[str]) -> pd.DataFrame:
        """
        Args:
            df: dataframe to transform
            columns: columns to convert to datetime

        Returns:
            the dataframe with the converted columns
        """
        for col in columns:
            df[col] = pd.to_datetime(df[col])
        return df

    @staticmethod
    def create_cxr_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            df: dataframe with StudyDate (YYYYMMDD) and StudyTime (HHMMSS[.fff]) columns

        Returns:
            the dataframe with an added StudyDateTime column
        """
        df['StudyDate'] = pd.to_datetime(df['StudyDate'].astype(str), format='%Y%m%d')
        df['StudyTime'] = df['StudyTime'].astype(str).str.split('.').str[0].str.zfill(6)  # Ensure HHMMSS format
        df['StudyDateTime'] = pd.to_datetime(df['StudyDate'].dt.strftime('%Y-%m-%d') + ' ' +
                                             df['StudyTime'].str[:2] + ':' +
                                             df['StudyTime'].str[2:4] + ':' +
                                             df['StudyTime'].str[4:], format='%Y-%m-%d %H:%M:%S')
        return df

    @staticmethod
    def impute_notes(patients_embed: pd.DataFrame,
                     empty_str_embed: pd.DataFrame,
                     list_of_features: List[str]) -> pd.DataFrame:
        """
        Replaces the missing note features of patients without notes by the embedding of an empty string.

        Args:
            patients_embed: patients features
            empty_str_embed: one-row dataframe with the embedding of an empty string
            list_of_features: note feature columns to impute

        Returns:
            patients features with imputed note features
        """
        # Get patients without notes, and replace Nan with the embedding of an empty string
        empty_str = patients_embed[list_of_features].isna().all(axis=1)
        patients_embed.loc[empty_str, list_of_features] = empty_str_embed.iloc[0].values.tolist()

        return patients_embed

    @staticmethod
    def sample_one_single_view(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
        """
        Keeps one image per group and image time, preferring AP then PA views.

        Args:
            df: images with ViewPosition and img_time columns
            id_cols: columns identifying a group (e.g. subject_id, hadm_id)

        Returns:
            one image per group and img_time
        """
        # Define view preferences as a dictionary
        view_priority = {'AP': 1, 'PA': 2}

        # Assign a priority score to each row, 999 for other views
        df['view_priority'] = df['ViewPosition'].map(view_priority).fillna(999)

        # Sort the dataframe so preferred views come first in each group
        df = df.sort_values(by=id_cols + ['img_time', 'view_priority'], ascending=True)

        # Drop duplicates to keep the first occurrence per group (the preferred one)
        df = df.drop_duplicates(subset=id_cols + ['img_time'], keep='first').reset_index(drop=True)

        return df
