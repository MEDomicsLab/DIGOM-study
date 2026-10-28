CREATE TABLE cxr_adm_ids AS
WITH icu_ids AS (
    SELECT DISTINCT subject_id, hadm_id, stay_id 
    FROM mimiciv_icu.icustays
),

adm_cxr AS (
    SELECT 
        a.subject_id, 
        i.hadm_id, 
        a.admittime, 
        a.edregtime, 
        a.dischtime, 
        i.stay_id
    FROM mimiciv_hosp.admissions a
    INNER JOIN icu_ids i ON a.hadm_id = i.hadm_id
    INNER JOIN public.cxr_metadata cxr_data ON a.subject_id = cxr_data.subject_id
    WHERE 
        TO_TIMESTAMP(
            CONCAT(
                CAST(cxr_data.StudyDate AS TEXT), ' ', 
                LPAD(SPLIT_PART(cxr_data.StudyTime, '.', 1), 6, '0')
            ), 
            'YYYYMMDD HH24MISS'
        )::TIMESTAMP WITHOUT TIME ZONE BETWEEN COALESCE(a.edregtime, a.admittime) AND a.dischtime
)
SELECT DISTINCT subject_id, hadm_id, stay_id FROM adm_cxr;


\COPY cxr_adm_ids TO 'csvs/utils/icu_modalities/cxr_adm_ids.csv' WITH CSV HEADER;
