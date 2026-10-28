"""
Filename: creation_icu_cohort.py

Author: Hakima Laribi

Description: Creation of the MIMIC-IV ICU cohort. Builds, for each ICU stay with a usable frontal (AP/PA) chest X-ray
taken in the first 24h after ICU admission, the lab, chart and procedure features, the image and radiology note
embeddings, the demographics and the 48h targets (mortality and discharge alive), 24h after ICU admission.
"""
import os
from typing import List, Literal

import numpy as np
import pandas as pd

from src.data.preprocessing import constants
from src.data.preprocessing.base_mimic_cohort import CohortMIMIC


class ICUCohortMIMIC(CohortMIMIC):
    """
    Create MIMIC ICU Cohort, containing demographics, lab, chart and procedure events, CXR embeddings and
    radiology notes embeddings, extracted 24h after ICU admission
    """
    DELTA_TIME = 24  # hours after ICU admission at which the features are extracted
    METRICS_COL = ['', 'present']
    ID_COLS = ['subject_id', 'hadm_id', 'stay_id', 'study_id', 'dicom_id']
    TIME_COLS = ['charttime', 'intime', 'dischtime', 'deathtime', 'StudyDateTime', 'admittime', 'edregtime',
                 'hospital_day']

    def __init__(self,
                 cxr_adm_ids: str,
                 cxr_meta_data: str,
                 admissions: str,
                 icustays: str,
                 radiology_notes: str):
        """
        Args:
            cxr_adm_ids: path to the csv of the cohort stays (subject_id, hadm_id, stay_id)
            cxr_meta_data: path to the MIMIC-CXR-JPG metadata csv
            admissions: path to the MIMIC-IV admissions csv
            icustays: path to the MIMIC-IV icustays csv
            radiology_notes: path to the MIMIC-IV-Note radiology csv
        """
        # Set protected attributes
        self._cxr_adm_ids = pd.read_csv(cxr_adm_ids, dtype={'stay_id': int, 'subject_id': int, 'hadm_id': int})
        self._cxr_meta_data = pd.read_csv(cxr_meta_data)
        self._admissions = pd.read_csv(admissions)
        self._icustays = pd.read_csv(icustays)
        self._radiology_notes = pd.read_csv(radiology_notes)

        # Transform time columns to date time
        self._admissions = self.transform_to_datetime(self._admissions, ['admittime', 'dischtime', 'edregtime'])
        self._icustays = self.transform_to_datetime(self._icustays, ['intime'])
        self._radiology_notes = self.transform_to_datetime(self._radiology_notes, ['charttime'])

        # Keep only cohort patients
        self._cxr_meta_data = self._cxr_meta_data[
            self._cxr_meta_data['subject_id'].isin(self._cxr_adm_ids['subject_id'])]
        self._admissions = self._admissions[self._admissions['hadm_id'].isin(self._cxr_adm_ids['hadm_id'])]
        self._icustays = self._icustays[self._icustays['stay_id'].isin(self._cxr_adm_ids['stay_id'])]
        self._radiology_notes = self._radiology_notes[
            self._radiology_notes['subject_id'].isin(self._cxr_adm_ids['subject_id'])]

        # Get ids and time of images included in the study
        self._imaging_ids = self.get_study_images()

    def __call__(self,
                 demographics: pd.DataFrame,
                 lab: pd.DataFrame,
                 chart: pd.DataFrame,
                 procedure: pd.DataFrame,
                 dense_features_path: List[str],
                 note_features_path: List[str],
                 saving_file: str,
                 empty_string_embed: pd.DataFrame):
        """
        Builds the cohort and saves icu_cxr_24h_mode.csv (one image per stay and image time) and
        icu_rad_demo_events_24h_mode.csv (events, radiology notes, demographics and targets) in saving_file.

        Args:
            demographics: demographics per admission
            lab: lab events
            chart: chart events
            procedure: procedure events
            dense_features_path: paths to the image embedding csvs
            note_features_path: paths to the radiology note embedding csvs
            saving_file: directory where the csvs are saved
            empty_string_embed: one-row dataframe with the embedding of an empty note
        """
        # First make sure all cohort admissions are in demographics
        if not self._cxr_adm_ids['hadm_id'].isin(demographics['hadm_id']).all():
            raise Exception('Demographics and ICU Cohorts do not match')

        # Get time series embeddings
        ts_embeddings = self.get_events_embeddings(lab, chart, procedure)

        # Get all images embeddings
        cxr_embeddings = self.get_cxr_embeddings(dense_features_path)
        cxr_embeddings = cxr_embeddings[['subject_id', 'hadm_id', 'stay_id', 'dicom_id', 'img_time', 'ViewPosition'] +
                                        [f'feature_{i}' for i in range(constants.N_IMG_FEAT_ICU)]]
        cxr_embeddings.to_csv(os.path.join(saving_file, 'icu_cxr_24h_mode.csv'), index=False)

        # Get notes embeddings
        rad_embeddings = self.get_rad_embeddings(note_features_path)

        # Concatenate embeddings
        patients_embed = pd.concat([ts_embeddings, rad_embeddings], axis=1).reset_index()

        patients_embed = self._cxr_adm_ids[['hadm_id', 'stay_id']].merge(patients_embed, on='stay_id', how='outer')
        # keep only those who had an image before the 24h after ICU transfer
        patients_embed = patients_embed[patients_embed['stay_id'].isin(cxr_embeddings['stay_id'].tolist())]

        # Impute NAN values in patients who didn't have a measurement of one or more type of events
        patients_embed = self.impute_events(patients_embed)

        # Get patients without notes, and replace Nan with the embedding of an empty string
        patients_embed = self.impute_notes(patients_embed,
                                           empty_string_embed,
                                           [f'rad_{i}' for i in range(constants.N_NOTE_FEAT)])

        # Add demographics
        patients_embed = self.get_demographics_targets(patients_embed, demographics)

        patients_embed.to_csv(os.path.join(saving_file, 'icu_rad_demo_events_24h_mode.csv'), index=False)

    def get_demographics_targets(self,
                                 patients_embed: pd.DataFrame,
                                 demographics: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            patients_embed: features with hadm_id and stay_id
            demographics: demographics per admission

        Returns:
            the valid stays with demographics and the disch_alive_48h and mortality_48h targets
        """
        # Add demographics
        patients_embed = demographics.merge(patients_embed, on='hadm_id', how='right')

        # Add targets and validity column
        id_cols = ['hadm_id', 'stay_id']
        targets = self.compute_targets(id_cols)
        patients_embed = patients_embed.merge(targets, on=id_cols, how='left')

        # keep only valid stays
        patients_embed = patients_embed[patients_embed['validity'] == 1].drop(columns='validity')

        return patients_embed

    def compute_targets(self, id_cols: List[str]) -> pd.DataFrame:
        """
        Computes the 48h targets.

        Args:
            id_cols: id columns to return with the targets

        Returns:
            id_cols with disch_alive_48h, mortality_48h and validity
        """
        # Get the corresponding time information of the icu stays of the study
        adm_icu = pd.merge(self._admissions, self._icustays, on=['subject_id', 'hadm_id'])
        adm_icu = adm_icu[adm_icu['stay_id'].isin(self._cxr_adm_ids['stay_id'])]

        time_col = 'intime'
        delta_time = self.DELTA_TIME

        adm_icu['mortality_48h'] = np.where(
            (adm_icu['deathtime'] > adm_icu[time_col] + pd.Timedelta(hours=48 + delta_time)) |
            adm_icu['deathtime'].isna(),
            0, 1)

        adm_icu['disch_alive_48h'] = np.where(
            (adm_icu['dischtime'] < (adm_icu[time_col] + pd.Timedelta(hours=48 + delta_time))) &
            adm_icu['deathtime'].isna(),
            1, 0)

        # Find non eligible samples
        adm_icu['validity'] = np.where((adm_icu['dischtime'] > adm_icu[time_col] + pd.Timedelta(delta_time)), 1, 0)

        return adm_icu[id_cols + ['disch_alive_48h', 'mortality_48h', 'validity']]

    @staticmethod
    def impute_events(patients_embed: pd.DataFrame) -> pd.DataFrame:
        """
        Fills with 0 the procedure sums and the lab/chart presence columns of stays without these events.

        Args:
            patients_embed: features

        Returns:
            the features with imputed columns
        """
        ts_columns = [f'{col}_sum' for col in constants.PROCEDURE_ICU]
        # for labs and charts, we only impute the presence column with 0
        ts_columns += [f'{col}_present' for col in constants.LAB_ICU + constants.CHART_ICU]

        # impute the columns where it's Nan
        patients_embed[ts_columns] = patients_embed[ts_columns].fillna(0)

        return patients_embed

    def get_events_embeddings(self,
                              lab: pd.DataFrame,
                              chart: pd.DataFrame,
                              procedure: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            lab: lab events
            chart: chart events
            procedure: procedure events

        Returns:
            concatenated procedure, chart and lab features of the cohort stays, indexed by stay_id
        """
        ts_event = {}
        for df_events, event in zip([procedure, chart, lab], ['procedure', 'chart', 'lab']):
            print(f'Processing time series of {event} events')
            # Filter
            df_events = self.filter_events(df_events, 'charttime')
            df_events = df_events[df_events['stay_id'].isin(self._cxr_adm_ids['stay_id'])]
            ts_event[event] = self.get_ts_embeddings(df_events, event_group=event)

        # Concatenate time series measurements of all events
        ts_df = pd.concat(ts_event.values(), axis=1)

        return ts_df

    def filter_events(self,
                      df_events: pd.DataFrame,
                      time_column: str) -> pd.DataFrame:
        """
        Keeps the events that happened before 24h after ICU admission, for stays still in the hospital at that time.

        Args:
            df_events: events with time_column, intime and dischtime columns
            time_column: time column of the events

        Returns:
            the filtered events
        """
        # Convert data types
        df_events = self.transform_to_datetime(df_events, [time_column, 'intime', 'dischtime'])

        extraction_time = df_events['intime'] + pd.Timedelta(hours=self.DELTA_TIME)
        df_events = df_events[(df_events['dischtime'] > extraction_time) & (df_events[time_column] <= extraction_time)]

        return df_events

    @staticmethod
    def compute_custom_metrics(series: pd.Series, event_group: str) -> pd.Series:
        """
        Computes the metrics of one event over time.

        Args:
            series: values of the event, sorted by time
            event_group: 'lab', 'chart' or 'procedure'

        Returns:
            procedures: {'sum'}; labs and charts: {'' (last value), 'present'}
        """
        series = series.dropna()

        metrics = {}
        if event_group == 'procedure':
            metrics['sum'] = series.sum() if len(series) > 0 else 0
        else:  # chart or lab event
            metrics[''] = series.iloc[-1] if len(series) > 0 else np.nan
            metrics['present'] = 1 if len(series) > 0 else 0

        return pd.Series(metrics)

    def aggregate_stays(self, df_group: pd.DataFrame, event_group: str) -> pd.DataFrame:
        """
        Computes the metrics of each stay of an admission, using the events of the stay and of the previous stays.

        Args:
            df_group: events of one admission with a stay_id column
            event_group: 'chart' or 'procedure'

        Returns:
            metrics indexed by (stay_id, metric)
        """
        # Get measurement columns relative to a lab, chart or procedure event
        event_columns = [column for column in df_group.columns if column not in self.ID_COLS + self.TIME_COLS]
        df_group = df_group.reset_index()

        # Get unique stay ids in the current admission
        stay_ids = df_group['stay_id'].unique()

        # Create MultiIndex dataframe
        metrics = self.METRICS_COL if event_group != 'procedure' else ['sum']
        multi_index = pd.MultiIndex.from_product([stay_ids, metrics], names=['stay_id', 'metric'])
        df = pd.DataFrame(index=multi_index, columns=event_columns)

        if len(stay_ids) > 1:
            for stay_id in stay_ids:  # get information of current stay and stays before
                indexes = df_group[df_group['stay_id'] == stay_id].index
                ts = df_group.iloc[:indexes[-1] + 1][event_columns].apply(
                    lambda x: self.compute_custom_metrics(x, event_group))
                df.loc[(stay_id, ts.index.values.tolist()), event_columns] = ts.values
        else:
            ts = df_group[event_columns].apply(lambda x: self.compute_custom_metrics(x, event_group))
            df.loc[(stay_ids[0], ts.index.values.tolist()), event_columns] = ts.values

        return df

    def get_ts_embeddings(self,
                          df_events: pd.DataFrame,
                          event_group: Literal['chart', 'lab', 'procedure']) -> pd.DataFrame:
        """
        Computes the metrics of each event: labs over each stay, charts and procedures over the admission up to
        each stay.

        Args:
            df_events: events of the cohort stays
            event_group: 'chart', 'lab' or 'procedure'

        Returns:
            metrics indexed by stay_id, with columns {event}_{metric}
        """
        # Sort events according to the time it was taken
        df_events = df_events.sort_values(by='charttime')

        # Get event columns
        event_columns = [column for column in df_events.columns if column not in self.TIME_COLS + self.ID_COLS]

        if event_group != 'lab':
            ts_df = df_events.groupby('hadm_id')[['stay_id'] + event_columns].apply(
                lambda x: self.aggregate_stays(x, event_group))
            ts_df = ts_df.droplevel(0)
            ts_df = ts_df.reset_index().pivot(index='stay_id', columns='metric')
        else:
            ts_df = (df_events.groupby(['stay_id'])[event_columns]
                     .apply(lambda x: x.apply(lambda y: self.compute_custom_metrics(y, event_group))))
            ts_df = ts_df.reset_index().pivot(index=['stay_id'], columns='level_1')

        # Flatten MultiIndex columns
        ts_df.columns = ['_'.join(col).strip('_') for col in ts_df.columns]

        return ts_df

    def get_cxr_embeddings(self, dense_features_path: List[str]) -> pd.DataFrame:
        """
        Args:
            dense_features_path: paths to the image embedding csvs

        Returns:
            embeddings of one image per stay and image time (AP preferred over PA), taken before 24h after
            ICU admission
        """
        print('Getting CXR embeddings...')
        # Get the dense features of the images included in our study
        df_img = self.read_dense_features(dense_features_path).drop(columns='pt_group')

        # Ungroup dicom_ids
        exploded_img_ids = self._imaging_ids.explode('dicom_id').reset_index(drop=True)

        # Merge based on dicom_id to get admission ids
        id_images = ['hadm_id', 'StudyDateTime', 'dicom_id']
        df_img = exploded_img_ids[id_images].merge(df_img, on='dicom_id', how='inner').rename(
            columns={'StudyDateTime': 'img_time'})

        # Get discharge and admission time columns from admission and icu
        df_img_adm = pd.merge(self._admissions[['hadm_id', 'dischtime']], df_img, how='inner', on='hadm_id')
        df_img_adm = pd.merge(self._icustays[['hadm_id', 'stay_id', 'intime']], df_img_adm, on='hadm_id', how='inner')

        # Filter patients still in ICU and images before extraction time
        df_img_adm = self.filter_events(df_img_adm, 'img_time')

        # Add position view
        df_img_adm = self._cxr_meta_data[['dicom_id', 'ViewPosition', 'study_id']].merge(df_img_adm, how='inner')

        # Sort images by time
        df_img_adm = df_img_adm.sort_values(by=['hadm_id', 'stay_id', 'img_time', 'study_id'],
                                            ascending=True).reset_index(drop=True)

        return self.sample_one_single_view(df_img_adm, ['subject_id', 'hadm_id', 'stay_id'])

    def get_rad_embeddings(self, note_features_path: List[str]) -> pd.DataFrame:
        """
        Args:
            note_features_path: paths to the radiology note embedding csvs

        Returns:
            radiology note embeddings written before 24h after ICU admission, averaged per stay,
            with columns renamed to rad_{i}
        """
        print('Getting radiology notes embeddings...')
        # read note embeddings
        df_note_adm = self.read_note_features(note_features_path, 'radiology')

        # Merge with ICU to get stay_ids and icu
        df_rad_embeddings = pd.merge(self._icustays, df_note_adm, how='inner', on=['subject_id', 'hadm_id'])

        # Filter patients still in ICU and notes before extraction time
        df_rad_embeddings = self.filter_events(df_rad_embeddings, 'charttime')

        note_features = [f'feature_{i}' for i in range(constants.N_NOTE_FEAT)]

        # sort notes by time
        df_rad_embeddings = df_rad_embeddings.sort_values(by=['hadm_id', 'stay_id', 'charttime'],
                                                          ascending=True).reset_index(drop=True)

        # Average the embeddings of all notes of each stay
        grouped_embed = df_rad_embeddings.groupby(['stay_id'])[note_features].mean()
        return grouped_embed.rename(columns={f'feature_{i}': f'rad_{i}' for i in range(len(note_features))})
