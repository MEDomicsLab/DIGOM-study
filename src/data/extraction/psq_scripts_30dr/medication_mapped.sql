\Copy (
WITH admission_cxr AS (
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
),
mapped_medications AS (
    SELECT  distinct on (p.subject_id, p.hadm_id, p.ndc, p.starttime, m.MED_THERAPEUTIC_CLASS_DESCRIPTION) 
	p.subject_id, p.hadm_id, p.ndc, p.starttime, m.MED_THERAPEUTIC_CLASS_DESCRIPTION
    FROM mimiciv_hosp.prescriptions p
    JOIN admission_cxr a ON p.hadm_id = a.hadm_id
    JOIN public.ndc2therap m ON p.ndc = LPAD(m.ndc_medication_code, 11, '0')
	WHERE m.med_therapeutic_class_description is not null
	AND p.starttime is not NULL
	AND (p.starttime BETWEEN COALESCE(a.edregtime, a.admittime) AND a.dischtime)
	ORDER BY  p.subject_id, p.hadm_id, p.starttime ASC 
)
SELECT *
FROM mapped_medications
) TO 'csvs/utils/30dr_modalities/medication.csv' WITH CSV HEADER;
