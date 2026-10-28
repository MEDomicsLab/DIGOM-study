\COPY (
SELECT 
    le.subject_id, 
    adm.hadm_id, 
    icu.stay_id, 
    le.charttime,
    icu.intime,
    adm.dischtime,
    adm.deathtime,
    MAX(CASE WHEN dlab.label = 'Glucose' THEN COALESCE(le.valuenum, 0) END) AS "Glucose",
    MAX(CASE WHEN dlab.label = 'Potassium' THEN COALESCE(le.valuenum, 0) END) AS "Potassium",
    MAX(CASE WHEN dlab.label = 'Sodium' THEN COALESCE(le.valuenum, 0) END) AS "Sodium",
    MAX(CASE WHEN dlab.label = 'Chloride' THEN COALESCE(le.valuenum, 0) END) AS "Chloride",
    MAX(CASE WHEN dlab.label = 'Creatinine' THEN COALESCE(le.valuenum, 0) END) AS "Creatinine",
    MAX(CASE WHEN dlab.label = 'Urea Nitrogen' THEN COALESCE(le.valuenum, 0) END) AS "Urea Nitrogen",
    MAX(CASE WHEN dlab.label = 'Bicarbonate' THEN COALESCE(le.valuenum, 0) END) AS "Bicarbonate",
    MAX(CASE WHEN dlab.label = 'Anion Gap' THEN COALESCE(le.valuenum, 0) END) AS "Anion Gap",
    MAX(CASE WHEN dlab.label = 'Hemoglobin' THEN COALESCE(le.valuenum, 0) END) AS "Hemoglobin",
    MAX(CASE WHEN dlab.label = 'Hematocrit' THEN COALESCE(le.valuenum, 0) END) AS "Hematocrit",
    MAX(CASE WHEN dlab.label = 'Magnesium' THEN COALESCE(le.valuenum, 0) END) AS "Magnesium",
    MAX(CASE WHEN dlab.label = 'Platelet Count' THEN COALESCE(le.valuenum, 0) END) AS "Platelet Count",
    MAX(CASE WHEN dlab.label = 'Phosphate' THEN COALESCE(le.valuenum, 0) END) AS "Phosphate",
    MAX(CASE WHEN dlab.label = 'White Blood Cells' THEN COALESCE(le.valuenum, 0) END) AS "White Blood Cells",
    MAX(CASE WHEN dlab.label = 'Calcium, Total' THEN COALESCE(le.valuenum, 0) END) AS "Calcium, Total",
    MAX(CASE WHEN dlab.label = 'MCH' THEN COALESCE(le.valuenum, 0) END) AS "MCH",
    MAX(CASE WHEN dlab.label = 'Red Blood Cells' THEN COALESCE(le.valuenum, 0) END) AS "Red Blood Cells",
    MAX(CASE WHEN dlab.label = 'MCHC' THEN COALESCE(le.valuenum, 0) END) AS "MCHC",
    MAX(CASE WHEN dlab.label = 'MCV' THEN COALESCE(le.valuenum, 0) END) AS "MCV",
    MAX(CASE WHEN dlab.label = 'RDW' THEN COALESCE(le.valuenum, 0) END) AS "RDW",
    MAX(CASE WHEN dlab.label = 'Neutrophils' THEN COALESCE(le.valuenum, 0) END) AS "Neutrophils",
    MAX(CASE WHEN dlab.label = 'Vancomycin' THEN COALESCE(le.valuenum, 0) END) AS "Vancomycin"
FROM mimiciv_hosp.labevents le
JOIN mimiciv_hosp.admissions adm 
    ON le.subject_id = adm.subject_id 
    AND (le.charttime BETWEEN COALESCE(adm.edregtime, adm.admittime) AND adm.dischtime 
    OR le.hadm_id = adm.hadm_id)
JOIN mimiciv_icu.icustays icu 
    ON adm.hadm_id = icu.hadm_id
JOIN public.cxr_adm_ids hk 
    ON icu.stay_id = hk.stay_id
JOIN mimiciv_hosp.d_labitems dlab 
    ON le.itemid = dlab.itemid
WHERE dlab.label IN ('Glucose', 'Potassium', 'Sodium', 'Chloride', 'Creatinine',
                   'Urea Nitrogen', 'Bicarbonate', 'Anion Gap', 'Hemoglobin', 'Hematocrit',
                   'Magnesium', 'Platelet Count', 'Phosphate', 'White Blood Cells',
                   'Calcium, Total', 'MCH', 'Red Blood Cells', 'MCHC', 'MCV', 'RDW', 
                   'Neutrophils', 'Vancomycin')
GROUP BY le.subject_id, adm.hadm_id, icu.stay_id, le.charttime
HAVING COUNT(*) > 0
) TO 'csvs/utils/icu_modalities/labevents.csv' WITH CSV HEADER;
