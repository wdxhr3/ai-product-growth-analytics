WITH signup AS (
    SELECT
        user_id,
        DATE_TRUNC('month', signup_date::DATE) AS cohort_month,
        signup_date::DATE AS signup_date
    FROM users
),
activity AS (
    SELECT DISTINCT
        user_id,
        event_date::DATE AS active_date
    FROM events
),
retention AS (
    SELECT
        s.cohort_month,
        DATE_DIFF('day', s.signup_date, a.active_date) AS days_since_signup,
        COUNT(DISTINCT s.user_id) AS retained_users
    FROM signup s
    JOIN activity a
        ON s.user_id = a.user_id
       AND a.active_date >= s.signup_date
       AND DATE_DIFF('day', s.signup_date, a.active_date) BETWEEN 0 AND 30
    GROUP BY 1,2
),
cohort_size AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT user_id) AS cohort_users
    FROM signup
    GROUP BY 1
)
SELECT
    r.cohort_month,
    r.days_since_signup,
    r.retained_users,
    c.cohort_users,
    r.retained_users * 1.0 / c.cohort_users AS retention_rate
FROM retention r
JOIN cohort_size c USING (cohort_month)
ORDER BY 1,2;
