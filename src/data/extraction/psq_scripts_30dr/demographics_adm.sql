\Copy (
WITH 
admission_cxr AS (
    SELECT DISTINCT
	    a.subject_id, 
		a.hadm_id, 
		EXTRACT(YEAR FROM a.admittime) - pat.anchor_year + pat.anchor_age as age,
		pat.gender,
		a.insurance, 
		a.language, 
		a.marital_status, 
		a.race,
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
readmissions AS (
        SELECT 
            f.*,
            a2.admittime - f.dischtime AS readmission_gap,
            a2.hadm_id AS readmission_id,
            ROW_NUMBER() OVER (
                PARTITION BY f.subject_id, f.hadm_id 
                ORDER BY a2.admittime - f.dischtime
            ) AS rn
        FROM admission_cxr f
        LEFT JOIN mimiciv_hosp.admissions a2
            ON a2.subject_id = f.subject_id
            AND a2.admittime > f.dischtime 
            AND a2.admission_type NOT IN ('AMBULATORY OBSERVATION', 'EU OBSERVATION', 'DIRECT OBSERVATION')
    ),
readmission_info_all AS (
    SELECT 
		
		readmissions.*, 
        CASE 
            WHEN readmission_gap IS NOT NULL AND EXTRACT(DAY FROM readmission_gap) <= 30 THEN 'True'
            WHEN (deathtime IS NOT NULL AND (deathtime - admittime) <= INTERVAL '30 days') 
                 OR (discharge_location = 'DIED') THEN 'True'
				ELSE 'False'
        END AS readmitted_within_30days
    FROM readmissions
    WHERE rn = 1 OR readmission_id IS NULL
)
SELECT *
FROM readmission_info_all
) TO 'csvs/utils/30dr_modalities/demo_target.csv' WITH CSV HEADER;
