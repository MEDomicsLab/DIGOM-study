\COPY (
WITH admission_cxr AS (
    SELECT DISTINCT
        a.subject_id, 
        a.hadm_id, 
        a.admittime, 
        a.edregtime, 
        a.dischtime, 
        COALESCE(a.edregtime, a.admittime) AS start_time,
        a.admission_type, 
        a.discharge_location, 
        a.deathtime
    FROM mimiciv_hosp.admissions a
    JOIN public.cxr_metadata c ON a.subject_id = c.subject_id
    JOIN mimiciv_hosp.patients pat ON a.subject_id = pat.subject_id
    WHERE 
        TO_TIMESTAMP(
            CONCAT(
                CAST(c.studydate AS TEXT), ' ', 
                LPAD(SPLIT_PART(c.studytime, '.', 1), 6, '0')
            ), 
            'YYYYMMDD HH24MISS'
        )::TIMESTAMP WITHOUT TIME ZONE 
        BETWEEN COALESCE(a.edregtime, a.admittime) AND a.dischtime
    AND a.admission_type NOT IN ('AMBULATORY OBSERVATION', 'EU OBSERVATION', 'DIRECT OBSERVATION')
    AND (a.discharge_location NOT IN ('ACUTE HOSPITAL', 'HEALTHCARE FACILITY', 'AGAINST ADVICE') OR a.discharge_location IS NULL)
),
admission_lab AS (
    SELECT 
        lab.subject_id, 
        a.hadm_id, 
        lab.charttime, 
        lab.valuenum, 
        CONCAT(i.label, ' ', i.fluid) AS label_fluid, 
        lab.flag,
        a.start_time,
        FLOOR(EXTRACT(EPOCH FROM (lab.charttime - a.start_time)) / 86400)::INT + 1 AS hospital_day
    FROM mimiciv_hosp.labevents lab
    JOIN admission_cxr a 
        ON lab.subject_id = a.subject_id 
        AND (lab.charttime BETWEEN a.start_time AND a.dischtime)
    JOIN mimiciv_hosp.d_labitems i ON i.itemid = lab.itemid
    WHERE CONCAT(i.label, ' ', i.fluid) IN (
        'pH Other Body Fluid', 'Lymphocytes Joint Fluid', 'Hemoglobin Blood', 'Basophils Joint Fluid', 'Chloride Blood', 
        'Glucose Blood', 'Monocytes Blood', 'Eosinophils Joint Fluid', 'pH Urine', 'Monocytes Ascites', 'Basophils Cerebrospinal Fluid', 
        'Hematocrit Urine', 'Bicarbonate Blood', 'Basophils Other Body Fluid', 'Potassium Blood', 'Eosinophils Blood', 'Eosinophils Pleural', 'L Blood', 
        'Lymphocytes Ascites', 'Creatinine Blood', 'Eosinophils Ascites', 'pO2 Blood', 'Lymphocytes Other Body Fluid', 'Hematocrit Blood', 'Leukocytes Urine', 
        'Calcium, Total Blood', 'Anion Gap Blood', 'Eosinophils Other Body Fluid', 'Sodium Blood', 'Lymphocytes Blood', 'Basophils Pleural', 'Lactate Blood', 'H Blood', 
        'Neutrophils Blood', 'Monocytes Cerebrospinal Fluid', 'Troponin T Blood', 'Platelet Count Blood', 'Basophils Blood', 'Monocytes Joint Fluid', 'Eosinophils Urine', 
        'pH Blood', 'Glucose Urine', 'Eosinophils Cerebrospinal Fluid', 'pCO2 Blood', 'Basophils Ascites', 'Lymphocytes Pleural'
    )
    AND lab.valuenum IS NOT NULL
),
last_lab_values AS (
    SELECT 
        subject_id,
        hadm_id,
        label_fluid,
        hospital_day,
        charttime,
        valuenum,
        flag,
        ROW_NUMBER() OVER (
            PARTITION BY subject_id, hadm_id, label_fluid, hospital_day
            ORDER BY charttime DESC
        ) AS rn
    FROM admission_lab
)
SELECT 
    l.subject_id,
    l.hadm_id,
    l.label_fluid,
    l.hospital_day,
    l.charttime,
    l.valuenum AS last_value,
    COALESCE(l.flag, 'normal') AS last_flag
FROM last_lab_values l
WHERE l.rn = 1
ORDER BY l.subject_id, l.hadm_id,l.hospital_day, l.label_fluid
) TO 'csvs/utils/30dr_modalities/labs_time.csv' WITH CSV HEADER;