\COPY (
SELECT 
DISTINCT
	hk.subject_id,
	hk.hadm_id,
	EXTRACT(YEAR FROM adm.admittime) - pat.anchor_year + pat.anchor_age as age,
    pat.gender, 
    adm.insurance, 
    adm.language, 
    adm.marital_status, 
    adm.race
FROM mimiciv_hosp.patients pat 
JOIN public.cxr_adm_ids hk
ON pat.subject_id = hk.subject_id
JOIN mimiciv_hosp.admissions adm 
ON adm.hadm_id = hk.hadm_id
) TO 'csvs/utils/icu_modalities/demographics.csv' WITH CSV HEADER;
