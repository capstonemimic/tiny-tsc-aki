.read typed_mimic.sql

--- cohort extraction -------
SET VARIABLE obs_window = 24::DOUBLE;

COPY (
    WITH icu_cohort AS (
        SELECT 
            ie.stay_id,
            ie.hadm_id,
            ie.subject_id,
            ie.intime::TIMESTAMP AS intime,
            ie.outtime::TIMESTAMP AS outtime,
            EXTRACT(EPOCH FROM (ie.outtime::TIMESTAMP - ie.intime::TIMESTAMP)) / 3600.0 AS icu_los_hours
        FROM icustays ie
        INNER JOIN patients p ON ie.subject_id = p.subject_id
        WHERE 
            -- Adults only (age >= 18)
            (p.anchor_age::INTEGER + EXTRACT(YEAR FROM ie.intime::TIMESTAMP) - p.anchor_year::INTEGER) >= 18
            -- ICU stay >= one observation window (obs_window = 24 hours)
            AND EXTRACT(EPOCH FROM (ie.outtime::TIMESTAMP - ie.intime::TIMESTAMP)) / 3600.0 >= CAST(getvariable('obs_window') AS DOUBLE)
    ),

    data_subset AS (
        select u.*, c.subject_id from ugly_data_table u inner join icu_cohort c on u.stay_id = c.stay_id
    ),

    -- don't filter out features
    features AS (
        select d.*, FLOOR(d.timedelta / getvariable('obs_window')) as window_index,
        (d.timedelta - window_index * getvariable('obs_window')) as timedelta_window
        from data_subset d where d.measurement_type != 'aki_kdigo' -- and timedelta <= getvariable('obs_window')
    ),

    target AS (
        -- window_index is the window immediately preceding the one in which
        -- AKI was first detected (the "predictive" window). When AKI is
        -- detected within the very first observation window (window 0),
        -- this would compute to -1: there is no valid prior window to
        -- predict from, and joining on window_index = -1 previously matched
        -- coincidental pre-ICU-admission data (negative timedelta), leaking
        -- observations at or after aki_time into the "predictive" window.
        -- Such stays are excluded (window_index left NULL, which cannot
        -- match any real feature window) rather than mislabeled.
        select stay_id, hadm_id,
            MIN(timedelta) FILTER (WHERE measurement_type='aki_kdigo' AND valuenum = 1) as aki_time,
            CASE
                WHEN (FLOOR(aki_time / getvariable('obs_window')) - 1) >= 0
                THEN (FLOOR(aki_time / getvariable('obs_window')) - 1)
                ELSE NULL
            END as window_index
        from data_subset group by stay_id, hadm_id
    ),
    pivoted_features AS (
        PIVOT features ON measurement_type
        USING list(struct_pack(t := timedelta_window, v := valuenum) order by timedelta ASC)
        GROUP BY stay_id, hadm_id, subject_id, window_index
    )
    
    select p.*, t.aki_time, (p.window_index = t.window_index) AS target from pivoted_features p inner join target t on p.stay_id = t.stay_id
        where p.window_index <= t.window_index
    union
    select p.*, t.aki_time, FALSE AS target from pivoted_features p left join target t on p.stay_id = t.stay_id
    where t.aki_time IS NULL
    
) TO './cohort-windowed.parquet' (FORMAT parquet);
