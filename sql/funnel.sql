WITH user_stage AS (
    SELECT
        u.user_id,
        TRUE AS registered,
        u.activated_24h,
        COUNT(DISTINCT c.conversation_id) > 0 AS used_ai_workflow,
        COUNT(DISTINCT CASE WHEN c.is_advanced_feature THEN c.conversation_id END) > 0 AS used_advanced_feature,
        COUNT(DISTINCT CASE WHEN e.event_name = 'billing_viewed' THEN e.event_id END) > 0 AS viewed_billing,
        COUNT(DISTINCT CASE WHEN e.event_name = 'subscribe_clicked' THEN e.event_id END) > 0 AS clicked_subscribe,
        u.paid_converted
    FROM users u
    LEFT JOIN conversations c ON u.user_id = c.user_id
    LEFT JOIN events e ON u.user_id = e.user_id
    GROUP BY 1,2,3,8
),
funnel AS (
    SELECT '01_registered' AS step, COUNT(*) AS users FROM user_stage
    UNION ALL
    SELECT '02_activated_24h', COUNT(*) FROM user_stage
    WHERE activated_24h
    UNION ALL
    SELECT '03_used_ai_workflow', COUNT(*) FROM user_stage
    WHERE activated_24h AND used_ai_workflow
    UNION ALL
    SELECT '04_used_advanced_feature', COUNT(*) FROM user_stage
    WHERE activated_24h AND used_ai_workflow AND used_advanced_feature
    UNION ALL
    SELECT '05_viewed_billing', COUNT(*) FROM user_stage
    WHERE activated_24h AND used_ai_workflow AND used_advanced_feature AND viewed_billing
    UNION ALL
    SELECT '06_clicked_subscribe', COUNT(*) FROM user_stage
    WHERE activated_24h AND used_ai_workflow AND used_advanced_feature AND viewed_billing AND clicked_subscribe
    UNION ALL
    SELECT '07_paid', COUNT(*) FROM user_stage
    WHERE activated_24h AND used_ai_workflow AND used_advanced_feature AND viewed_billing AND clicked_subscribe AND paid_converted
),
with_base AS (
    SELECT
        step,
        users,
        FIRST_VALUE(users) OVER (ORDER BY step) AS registered_users,
        LAG(users) OVER (ORDER BY step) AS previous_users
    FROM funnel
)
SELECT
    step,
    users,
    users * 1.0 / registered_users AS conversion_from_registered,
    users * 1.0 / NULLIF(previous_users, 0) AS conversion_from_previous
FROM with_base
ORDER BY step;
