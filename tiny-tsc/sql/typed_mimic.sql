-- no temporal filters just process the db into the required shape.
-- create a temporary table with admissions data.
create temporary view if not exists typed_admissions as (
    select
        subject_id,
        try_cast(hadm_id as integer) as hadm_id,
        try_cast(admittime as timestamp) as admittime, -- admission time
        try_cast(dischtime as timestamp) as dischtime, -- dischard time
        try_cast(deathtime as timestamp) as deathtime, -- death time
        admission_type,
        admission_location,
        discharge_location,
        marital_status,
        race,
        try_cast(edregtime as timestamp) as edregtime,
        try_cast(edouttime as timestamp) as edouttime,
        try_cast(hospital_expire_flag as integer) as expired_in_hospital,
        extract(
            epoch from (try_cast(dischtime as timestamp) - try_cast(admittime as timestamp))
        ) / 3600.0 as length_of_admission_hours
    from admissions
);

-- map item ids to feature names
create temporary table if not exists itemid_mapping AS (
    SELECT * FROM (VALUES
        -- Chart Events: Blood Pressure
        (220051, 'arterial_blood_pressure_diastolic'),
        (225310, 'arterial_blood_pressure_diastolic'),
        (225309, 'arterial_blood_pressure_systolic'),
        (220050, 'arterial_blood_pressure_systolic'),
        (220180, 'non_invasive_blood_pressure_diastolic'),
        (220179, 'non_invasive_blood_pressure_systolic'),
        -- Chart Events: pH (kept separate by specimen/fluid type)
        (223830, 'ph_arterial'),
        (220274, 'ph_venous'),
        (220734, 'ph_dipstick'),
        -- Chart Events: Capillary Refill
        (223951, 'capillary_refill_r'),
        (224308, 'capillary_refill_l'),
        -- Chart Events: Glasgow Coma Scale
        (220739, 'gcs_eyes'),
        (223900, 'gcs_verbal'),
        (223901, 'gcs_motor'),
        -- Chart Events: Respiratory
        (228640, 'etco2'),
        (223835, 'fio2'),  -- Primary FiO2 from bedside monitor (% or fraction)
        (220277, 'o2_saturation_pulseoxymetry'),
        (220210, 'respiratory_rate'),
        (224690, 'respiratory_rate'),
        -- Chart Events: Glucose (VERIFIED: both serum/blood, safe to merge)
        (220621, 'glucose_serum'),
        (226537, 'glucose_serum'),
        -- Chart Events: Vitals
        (220045, 'heart_rate'),
        (223762, 'temperature_celsius'),
        (223761, 'temperature_celsius'),  -- Will convert F to C
        -- Chart Events: Lab values
        (220615, 'creatinine_serum'),  -- Merges with lab creatinine (serum)
        (220545, 'hematocrit_serum'),  -- Merges with lab hematocrit (serum)
        (220546, 'wbc'),               -- Blood WBC, stays separate from urine WBC
        -- Lab Events: pH (FIXED: split by specimen/fluid - arterial vs other-fluid vs urine)
        (50820, 'ph_arterial'),  -- Blood, avg 7.37 - true arterial
        (50831, 'ph_other_fluid'),  -- fluid='Other Body Fluid' per d_labitems, avg 7.40 - NOT confirmed arterial
        (51491, 'ph_urine'),  -- Urine, avg 6.21
        (51094, 'ph_urine'),  -- Urine, avg 4.84
        (52730, 'ph_urine'),  -- Urine, avg 6.16
        -- Lab Events: Glucose (VERIFIED: 50931/50809/52569 are blood glucose)
        (50931, 'glucose'),    -- 3.6M measurements, avg 127 mg/dL
        (50809, 'glucose'),    -- 280K measurements, avg 149 mg/dL
        (52569, 'glucose'),    -- 1.3K measurements, avg 124 mg/dL
        (51478, 'glucose1'),   -- 63K measurements, avg 444 mg/dL (likely CSF)
        (51981, 'glucose2'),   -- 129 measurements, avg 263 mg/dL (rare)
        -- Lab Events: Creatinine (FIXED: serum lab itemids now share the
        -- 'creatinine_serum' tag so they actually merge with chart 220615;
        -- urine MUST stay separate)
        (50912, 'creatinine_serum'),  -- 4.3M, avg 1.34 mg/dL (SERUM) - was 'creatinine'
        (51081, 'creatinine_serum'),  -- 714, avg 1.27 mg/dL (SERUM)
        (52546, 'creatinine_serum'),  -- 1.2K, avg 1.00 mg/dL (SERUM) - was 'creatinine'
        (51082, 'creatinine_urine'),  -- 237K, avg 102 mg/dL (URINE!)
        -- Lab Events: Hematocrit (FIXED: serum lab itemids now share the
        -- 'hematocrit_serum' tag so they actually merge with chart 220545)
        (51221, 'hematocrit_serum'),      -- 4.3M, avg 33.3% - was 'hematocrit'
        (51638, 'hematocrit_serum'),      -- 1.2K, avg 40.5% - was 'hematocrit'
        (51639, 'hematocrit_serum'),      -- 17, avg 39.5% - was 'hematocrit'
        (51480, 'hematocrit0'),     -- 11, avg 8.5% (abnormal, keep separate)
        -- Lab Events: Blood gases
        (50818, 'pco2'),
        (50821, 'po2'),
        -- Lab Events: Sodium (VERIFIED: same range, safe to merge)
        (50983, 'sodium'),  -- 4.1M, avg 138.5 mEq/L
        (52623, 'sodium'),  -- 1.3K, avg 138.4 mEq/L
        -- Lab Events: WBC (VERIFIED: 51516 is URINE WBC, not blood!)
        (51516, 'wbc_urine'),  -- Units: #/hpf (urine), NOT blood WBC
        -- Lab Events: BUN
        (51006, 'bun'),  -- mg/dL
        -- Lab Events: Anion Gap
        (50868, 'anion_gap'),  -- mEq/L
        -- Lab Events: FiO2 from blood gas
        (50816, 'fio2_lab')  -- % (from blood gas analysis)
    ) AS t(itemid, measurement_type)
);

-- typed chartevents
-- bulk of the data is sourced from ICU charts
create temporary view if not exists typed_chartevents as (
    select
        c.subject_id,
        cast(c.hadm_id as integer) as hadm_id,
        cast(c.stay_id as integer) as stay_id,
        itm.measurement_type,
        c.valueuom,
        cast(c.charttime as timestamp) as abs_timestamp,
        try_cast(c.valuenum as double) as try_numeric,
        case
         -- convert capillary refill to binary feature
        when c.itemid in (223951, 224308) then
            case
                when upper(c.value) like '%ABNORMAL%' then 1.0
                when upper(c.value) like '%NORMAL%' then 0.0
            end
         -- convert to celsius
         when c.itemid = 223761 then
            (try_numeric - 32) * 5.0/9.0
        
          when c.itemid = 223835 then 
          case
            when try_numeric <= 1 then try_numeric * 100 -- restore percentage from fraction
            when try_numeric > 100 then NULL -- drop illegal value
            else try_numeric -- known index_scan_percentage
          end
         when c.itemid = 223900 and c.value = 'No Response-ETT' then 0.0
         else try_numeric
        end as valuenum
    from chartevents c
    inner join itemid_mapping itm on itm.itemid = c.itemid
    where c.valuenum is not NULL
);

-- typed labevents
create temporary view if not exists typed_labevents as (
    select
        l.subject_id,
        cast(l.hadm_id as integer) as hadm_id,
        itm.measurement_type,
        l.valueuom,
        try_cast(l.charttime as timestamp) as abs_timestamp,
        coalesce(
            try_cast(l.valuenum as double),
            case
                when l.value is null or l.value in ('', '---', '--') then null
                when upper(l.value) = 'NEG' then 0.0
                else 1.0
            end
        ) as valuenum
    from labevents l
    inner join itemid_mapping itm on try_cast(l.itemid as integer) = itm.itemid
);

-- typed outputevents
-- no itemid_mapping join available here, so the itemid list is the only
-- selectivity constraint (not redundant the way it was for labevents)
create temporary view if not exists typed_outputevents as (
    select
        oe.subject_id,
        cast(oe.hadm_id as integer) as hadm_id,
        cast(oe.stay_id as integer) as stay_id,
        'urine_output' as measurement_type,
        oe.valueuom,
        try_cast(oe.charttime as timestamp) as abs_timestamp,
        -- itemid 227488 = GU Irrigant Volume In: negate positive values so
        -- irrigant fluid going in is subtracted from total urine output
        case
            when try_cast(oe.itemid as integer) = 227488 and try_cast(oe.value as double) > 0
            then -1 * try_cast(oe.value as double)
            else try_cast(oe.value as double)
        end as valuenum
    from outputevents oe
    where try_cast(oe.itemid as integer) in (
        226559, 226560, 226561, 226584, 226563, 226564,
        226565, 226567, 226557, 226558, 227488, 227489
    )
);

-- gcs: raw component pulls
create temporary view if not exists typed_gcs_components_raw as (
    select
        cast(c.hadm_id as integer) as hadm_id,
        cast(c.stay_id as integer) as stay_id,
        try_cast(c.charttime as timestamp) as abs_timestamp,
        itm.measurement_type,
        case
            when try_cast(c.itemid as integer) = 223900 and c.value = 'No Response-ETT' then 0.0
            else try_cast(c.valuenum as double)
        end as valuenum
    from chartevents c
    inner join itemid_mapping itm on try_cast(c.itemid as integer) = itm.itemid
    where itm.measurement_type in ('gcs_motor', 'gcs_verbal', 'gcs_eyes')
);

-- gcs: collapse to one row per exact timestamp
create temporary view if not exists typed_gcs_components as (
    select
        stay_id, hadm_id, abs_timestamp,
        max(case when measurement_type = 'gcs_motor' then valuenum end) as gcs_motor,
        max(case when measurement_type = 'gcs_verbal' then valuenum end) as gcs_verbal,
        max(case when measurement_type = 'gcs_eyes' then valuenum end) as gcs_eyes
    from typed_gcs_components_raw
    group by stay_id, hadm_id, abs_timestamp
);

-- gcs: 6-hour forward-fill per component, in absolute time
create temporary view if not exists typed_gcs_filled as (
    select
        stay_id, hadm_id, abs_timestamp,
        coalesce(
            gcs_motor,
            last_value(gcs_motor ignore nulls) over (
                partition by stay_id order by abs_timestamp
                range between interval '6 hours' preceding and current row)
        ) as gcs_motor_filled,
        coalesce(
            gcs_verbal,
            last_value(gcs_verbal ignore nulls) over (
                partition by stay_id order by abs_timestamp
                range between interval '6 hours' preceding and current row)
        ) as gcs_verbal_filled,
        coalesce(
            gcs_eyes,
            last_value(gcs_eyes ignore nulls) over (
                partition by stay_id order by abs_timestamp
                range between interval '6 hours' preceding and current row)
        ) as gcs_eyes_filled
    from typed_gcs_components
);

-- gcs: total, with ett fix (verbal imputed as 1 "unable to assess" rather  than forcing the total to the best possible score)
create temporary view if not exists typed_gcs_total as (
    select
        stay_id, hadm_id, abs_timestamp,
        'gcs_total' as measurement_type,
        coalesce(gcs_motor_filled, 6)
            + case when gcs_verbal_filled = 0 then 1 else coalesce(gcs_verbal_filled, 5) end
            + coalesce(gcs_eyes_filled, 4) as valuenum
    from typed_gcs_filled
    where gcs_motor_filled is not null or gcs_verbal_filled is not null or gcs_eyes_filled is not null
);

-- aki: raw creatinine readings, whole stay, no window (hadm_id added for
-- consistency with the other typed_* views so this can union alongside them)
create temporary view if not exists typed_creatinine_readings as (
    select
        cast(c.hadm_id as integer) as hadm_id,
        cast(c.stay_id as integer) as stay_id,
        try_cast(c.charttime as timestamp) as abs_timestamp,
        try_cast(c.valuenum as double) as creat
    from chartevents c
    where try_cast(c.itemid as integer) = 220615
);

-- aki: per-reading KDIGO signals, purely backward-looking (both window
-- functions only see the current row and earlier ones, so this can't leak
-- future information into a given timestamp's own flag)
create temporary view if not exists typed_creatinine_flagged as (
    select
        hadm_id,
        stay_id,
        abs_timestamp,
        creat,
        min(creat) over (
            partition by stay_id order by abs_timestamp
            range between interval '48 hours' preceding and current row
        ) as rolling_min_48h,
        min(creat) over (
            partition by stay_id order by abs_timestamp
            rows between unbounded preceding and current row
        ) as running_min_creat
    from typed_creatinine_readings
);

-- rename for clarity: this is the creatinine-only component signal now,
-- not the final feature (was previously called typed_aki_kdigo)
create temporary view if not exists typed_aki_kdigo_creat as (
    select
        hadm_id,
        stay_id,
        abs_timestamp,
        'aki_kdigo_creat' as measurement_type,
        case
            when (creat - rolling_min_48h >= 0.3) or (creat >= 1.5 * running_min_creat)
            then 1.0
            else 0.0
        end as valuenum
    from typed_creatinine_flagged
);

-- weight: admission weight preferred, earliest daily weight as fallback,
-- one value per stay used as a fixed normalizer for the whole stay
create temporary view if not exists typed_weight as (
    select stay_id, weight_kg
    from (
        select
            cast(c.stay_id as integer) as stay_id,
            try_cast(c.valuenum as double) as weight_kg,
            row_number() over (
                partition by cast(c.stay_id as integer)
                order by
                    case when try_cast(c.itemid as integer) = 226512 then 0 else 1 end,
                    try_cast(c.charttime as timestamp) asc
            ) as rn
        from chartevents c
        where try_cast(c.itemid as integer) in (226512, 224639)
          and try_cast(c.valuenum as double) is not null
          and try_cast(c.valuenum as double) > 0
    )
    where rn = 1
);

-- rolling urine output sums + actual observed span per window, per event
create temporary view if not exists typed_urine_output_rolling as (
    select
        stay_id,
        hadm_id,
        abs_timestamp,
        weight_kg,
        sum(valuenum) over w6  as uo_sum_6h,
        sum(valuenum) over w12 as uo_sum_12h,
        sum(valuenum) over w24 as uo_sum_24h,
        extract(epoch from (abs_timestamp - min(abs_timestamp) over w6))  / 3600.0 as span_hours_6h,
        extract(epoch from (abs_timestamp - min(abs_timestamp) over w12)) / 3600.0 as span_hours_12h,
        extract(epoch from (abs_timestamp - min(abs_timestamp) over w24)) / 3600.0 as span_hours_24h
    from (
        select oe.*, w.weight_kg
        from typed_outputevents oe
        inner join typed_weight w on w.stay_id = oe.stay_id
    )
    window
        w6  as (partition by stay_id order by abs_timestamp range between interval '6 hours' preceding and current row),
        w12 as (partition by stay_id order by abs_timestamp range between interval '12 hours' preceding and current row),
        w24 as (partition by stay_id order by abs_timestamp range between interval '24 hours' preceding and current row)
);

-- kdigo urine-output stage>=1 flag, same shape as the other typed_* views
create temporary view if not exists typed_aki_kdigo_uo as (
    select
        hadm_id,
        stay_id,
        abs_timestamp,
        'aki_kdigo_uo' as measurement_type,
        case
            when weight_kg is null then null
            when span_hours_24h >= 24 and (uo_sum_24h / (weight_kg * 24.0)) < 0.3 then 1.0
            when span_hours_12h >= 12 and uo_sum_12h = 0 then 1.0  -- anuria proxy, see caveat above
            when span_hours_12h >= 12 and (uo_sum_12h / (weight_kg * 12.0)) < 0.5 then 1.0
            when span_hours_6h  >= 6  and (uo_sum_6h  / (weight_kg * 6.0))  < 0.5 then 1.0
            when span_hours_6h  >= 6  then 0.0
            else null
        end as valuenum
    from typed_urine_output_rolling
);

-- both component signals on one shared event timeline, one row per
-- (stay, timestamp) that either source actually reported at
create temporary view if not exists typed_aki_kdigo_events as (
    select stay_id, hadm_id, abs_timestamp,
        max(case when source = 'creat' then valuenum end) as creat_flag_raw,
        max(case when source = 'uo'    then valuenum end) as uo_flag_raw
    from (
        select stay_id, hadm_id, abs_timestamp, 'creat' as source, valuenum from typed_aki_kdigo_creat
        union all
        select stay_id, hadm_id, abs_timestamp, 'uo'    as source, valuenum from typed_aki_kdigo_uo
    )
    group by stay_id, hadm_id, abs_timestamp
);

-- forward-fill each component independently so every combined timestamp
-- carries the most recently known status of both criteria
create temporary view if not exists typed_aki_kdigo_filled as (
    select
        stay_id, hadm_id, abs_timestamp,
        last_value(creat_flag_raw ignore nulls) over (
            partition by stay_id order by abs_timestamp
            rows between unbounded preceding and current row
        ) as creat_flag,
        last_value(uo_flag_raw ignore nulls) over (
            partition by stay_id order by abs_timestamp
            rows between unbounded preceding and current row
        ) as uo_flag
    from typed_aki_kdigo_events
);

-- final merged feature: positive if EITHER criterion is positive
-- (definitive regardless of the other's status, since KDIGO is an "any
-- of" definition); negative only once BOTH criteria are known and both
-- are negative -- a single known-negative with the other still unknown
-- stays null rather than being called a confirmed negative
create temporary view if not exists typed_aki_kdigo as (
    select
        stay_id, hadm_id, abs_timestamp,
        'aki_kdigo' as measurement_type,
        case
            when creat_flag = 1.0 or uo_flag = 1.0 then 1.0
            when creat_flag = 0.0 and uo_flag = 0.0 then 0.0
            else null
        end as valuenum
    from typed_aki_kdigo_filled
);

create temporary view if not exists ugly_data_table as (
    with cte(stay_id, hadm_id, abs_timestamp, measurement_type, valuenum) as (
    select stay_id, hadm_id, abs_timestamp, measurement_type, valuenum from typed_chartevents
    union all
    select i.stay_id, l.hadm_id, l.abs_timestamp, l.measurement_type, l.valuenum from typed_labevents l inner join icustays i
    on i.hadm_id = l.hadm_id
    union all
    select stay_id, hadm_id, abs_timestamp, measurement_type, valuenum from typed_outputevents
    union all
    select stay_id, hadm_id, abs_timestamp, measurement_type, valuenum from typed_gcs_total
    union all
    select stay_id, hadm_id, abs_timestamp, measurement_type, valuenum from typed_aki_kdigo
    )
    select cte.stay_id, cte.hadm_id, epoch(cte.abs_timestamp - adm.admittime) / 3600 as timedelta, cte.measurement_type, cte.valuenum
    from cte inner join typed_admissions adm on cte.hadm_id = adm.hadm_id
);


--- cohort extraction -------
SET VARIABLE obs_window = 48::DOUBLE;

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
            -- ICU stay >= 48 hours
            AND EXTRACT(EPOCH FROM (ie.outtime::TIMESTAMP - ie.intime::TIMESTAMP)) / 3600.0 >= CAST(getvariable('obs_window') AS DOUBLE)
    ),

    data_subset AS (
        select u.* from ugly_data_table u inner join icu_cohort c on u.stay_id = c.stay_id
    ),

    features AS (
        select * from data_subset where measurement_type != 'aki_kdigo' and timedelta <= getvariable('obs_window')
    ),

    target AS (
        select stay_id, hadm_id, MIN(timedelta) FILTER (WHERE measurement_type='aki_kdigo' AND valuenum = 1) as aki_time from data_subset group by stay_id, hadm_id
    ),
    pivoted_features AS (
        PIVOT features ON measurement_type
        USING list(struct_pack(t := timedelta, v := valuenum) order by timedelta ASC)
        GROUP BY stay_id, hadm_id
    )

    select p.*, t.aki_time from pivoted_features p inner join target t on p.stay_id = t.stay_id
    where t.aki_time > getvariable('obs_window') OR t.aki_time IS NULL
) TO './cohort.parquet' (FORMAT parquet);
