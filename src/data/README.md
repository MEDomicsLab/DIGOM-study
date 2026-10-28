## Generation of the MIMIC-IV cohorts
This folder generates the cohorts of the three MIMIC-IV tasks: 30-day readmission risk (`30dr`), 48-hour ICU mortality risk (`mortality_48h`) and 48-hour ICU length of stay (`disch_alive_48h`). The two ICU tasks share the same cohort.

Run the four steps below in order, from the root of the repository. The other files of this folder are called by these steps and do not need to be run.

## 1. Get access to the data
Get access to [MIMIC-IV 3.1](https://physionet.org/content/mimiciv/), [MIMIC-CXR-JPG 2.1.0](https://physionet.org/content/mimic-cxr-jpg/) and [MIMIC-IV-Note 2.2](https://physionet.org/content/mimic-iv-note/) on PhysioNet (data use agreement and required training), then:
- download MIMIC-IV in `physionet.org/mimic-iv-3.1` and load it into a PostgreSQL database `mimiciv`, using the [mimic-code scripts](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv/buildmimic/postgres);
- download the MIMIC-CXR-JPG metadata (`mimic-cxr-2.0.0-metadata.csv`) in `physionet.org/image_features`, and load it into a `public.cxr_metadata` table of the same database;
- set your PhysioNet credentials, used to download the images and notes in step 3:
```
$ export PHYSIONET_USERNAME=<username>
$ export PHYSIONET_PASSWORD=<password>
```

## 2. Extract the EHR modalities
Run the SQL scripts in this order. They write the EHR modalities of each cohort in `csvs/utils/icu_modalities` and `csvs/utils/30dr_modalities`:
```
$ mkdir -p csvs/utils/icu_modalities csvs/utils/30dr_modalities
$ for f in extract_mm_ids extract_demographics extract_labevents_icu extract_chartevents extract_procedureevents; do
      psql -d mimiciv -f src/data/extraction/psql_scripts_48m_48los/$f.sql; done
$ for f in ncd2therap demographics_adm diagnoses procedures labs_time medication_mapped; do
      psql -d mimiciv -f src/data/extraction/psq_scripts_30dr/$f.sql; done
```

## 3. Extract the image and note embeddings
Run:
```
$ python -m src.data.extract_embeddings
```
It downloads the chest X-rays and embeds them with *DenseNet121-res224-chex* (in `physionet.org/image_features/cxr_features`), then downloads the radiology and discharge notes and embeds them with *Bio_ClinicalBERT* (in `physionet.org/note`). To save disk space, the images are downloaded and deleted one patient group at a time. A GPU is recommended.

## 4. Generate the cohorts
Check that the paths at the top of [generate_cohorts.py](generate_cohorts.py) match your files (in particular the number of note embedding files written in step 3), then run:
```
$ python -m src.data.generate_cohorts
```
It writes the cohorts in `csvs/datasets`, where the experiments read them: `30dr_cxr.csv` and `30dr_notes_ehr_demo.csv` for `30dr`, `icu_cxr_24h_mode.csv` and `icu_rad_demo_events_24h_mode.csv` for the ICU tasks.
