\COPY (
SELECT 
    pe.subject_id, 
    adm.hadm_id, 
    icu.stay_id, 
    pe.starttime AS charttime,  
    icu.intime,
    adm.dischtime,
    adm.deathtime,
    MAX(CASE WHEN di.label = 'Foley Catheter' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "Foley Catheter",
    MAX(CASE WHEN di.label = 'PICC Line' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "PICC Line",
    MAX(CASE WHEN di.label = 'Intubation' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "Intubation",
    MAX(CASE WHEN di.label = 'Peritoneal Dialysis' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "Peritoneal Dialysis",
    MAX(CASE WHEN di.label = 'Bronchoscopy' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "Bronchoscopy",
    MAX(CASE WHEN di.label = 'EEG' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "EEG",
    MAX(CASE WHEN di.label = 'Dialysis - CRRT' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "Dialysis - CRRT",
    MAX(CASE WHEN di.label = 'Dialysis Catheter' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "Dialysis Catheter",
    MAX(CASE WHEN di.label = 'Chest Tube Removed' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "Chest Tube Removed",
    MAX(CASE WHEN di.label = 'Hemodialysis' THEN CASE WHEN COALESCE(pe.value, 0) > 0 THEN 1 ELSE 0 END END) AS "Hemodialysis"
FROM mimiciv_icu.procedureevents pe
JOIN public.cxr_adm_ids hk
ON pe.stay_id = hk.stay_id
JOIN mimiciv_icu.icustays icu
ON pe.stay_id = icu.stay_id
JOIN mimiciv_icu.d_items di
ON pe.itemid = di.itemid
JOIN mimiciv_hosp.admissions adm
ON pe.hadm_id = adm.hadm_id 
WHERE di.label IN ('Foley Catheter', 'PICC Line', 'Intubation', 'Peritoneal Dialysis', 
                   'Bronchoscopy', 'EEG', 'Dialysis - CRRT', 'Dialysis Catheter', 
                   'Chest Tube Removed', 'Hemodialysis')
GROUP BY pe.subject_id, adm.hadm_id, icu.stay_id, pe.starttime
HAVING COUNT(*) > 0
) TO 'csvs/utils/icu_modalities/procedureevents.csv' WITH CSV HEADER;
