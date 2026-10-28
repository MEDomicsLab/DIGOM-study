\COPY (WITH admission_cxr AS (
    SELECT DISTINCT
	    a.subject_id, 
		a.hadm_id, 
        a.admittime, 
		a.edregtime, a.dischtime, a.admission_type, a.discharge_location, a.deathtime
    FROM mimiciv_hosp.admissions a
    JOIN public.cxr_metadata c 
    ON a.subject_id = c.subject_id
	JOIN mimiciv_hosp.patients pat
	ON a.subject_id = pat.subject_id
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
)
Select d.subject_id, d.hadm_id, d.icd_code, d.icd_version
FROM mimiciv_hosp.diagnoses_icd d
JOIN admission_cxr a ON a.hadm_id = d.hadm_id
) TO 'csvs/utils/30dr_modalities/diagnoses.csv' WITH CSV HEADER;