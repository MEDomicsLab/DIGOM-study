\COPY (
SELECT 
    ce.subject_id, 
    adm.hadm_id, 
    icu.stay_id, 
    ce.charttime,  
	icu.intime,
    adm.dischtime,
    adm.deathtime,
    MAX(CASE WHEN di.label = 'Heart Rate' THEN COALESCE(ce.valuenum, 0) END) AS "Heart Rate",
    MAX(CASE WHEN di.label = 'Non Invasive Blood Pressure systolic' THEN COALESCE(ce.valuenum, 0) END) AS "NIBP Systolic",
    MAX(CASE WHEN di.label = 'Non Invasive Blood Pressure diastolic' THEN COALESCE(ce.valuenum, 0) END) AS "NIBP Diastolic",
    MAX(CASE WHEN di.label = 'Non Invasive Blood Pressure mean' THEN COALESCE(ce.valuenum, 0) END) AS "NIBP Mean",
    MAX(CASE WHEN di.label = 'Respiratory Rate' THEN COALESCE(ce.valuenum, 0) END) AS "Respiratory Rate",
    MAX(CASE WHEN di.label = 'O2 saturation pulseoxymetry' THEN COALESCE(ce.valuenum, 0) END) AS "O2 Saturation",
    MAX(CASE WHEN di.label = 'GCS - Verbal Response' THEN COALESCE(ce.valuenum, 0) END) AS "GCS - Verbal",
    MAX(CASE WHEN di.label = 'GCS - Eye Opening' THEN COALESCE(ce.valuenum, 0) END) AS "GCS - Eye",
    MAX(CASE WHEN di.label = 'GCS - Motor Response' THEN COALESCE(ce.valuenum, 0) END) AS "GCS - Motor"
FROM mimiciv_icu.chartevents ce
JOIN public.cxr_adm_ids hk
ON ce.stay_id = hk.stay_id
JOIN mimiciv_icu.d_items di
ON ce.itemid = di.itemid
JOIN mimiciv_icu.icustays icu
ON ce.stay_id = icu.stay_id
JOIN mimiciv_hosp.admissions adm
ON ce.hadm_id = adm.hadm_id 
WHERE di.label IN ('Heart Rate', 'Non Invasive Blood Pressure systolic',
                   'Non Invasive Blood Pressure diastolic', 'Non Invasive Blood Pressure mean', 
                   'Respiratory Rate', 'O2 saturation pulseoxymetry', 
                   'GCS - Verbal Response', 'GCS - Eye Opening', 'GCS - Motor Response')
GROUP BY ce.subject_id, adm.hadm_id, icu.stay_id, ce.charttime
HAVING COUNT(*) > 0
) TO 'csvs/utils/icu_modalities/chartevents.csv' WITH CSV HEADER;
