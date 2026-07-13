WITH user_activity AS (
    SELECT
        u.user_id,
        u.signup_date,
        u.acquisition_channel,
        u.persona,
        u.company_size,
        u.activated_24h,
        u.paid_converted,
        u.current_plan,
        COUNT(DISTINCT e.event_id) AS total_events,
        COUNT(DISTINCT e.event_date) AS active_days,
        COUNT(DISTINCT c.conversation_id) AS conversations,
        SUM(c.total_tokens) AS total_tokens,
        SUM(c.cost_usd) AS model_cost_usd,
        AVG(c.response_latency_ms) AS avg_latency_ms,
        1 - AVG(CASE WHEN c.is_success THEN 1 ELSE 0 END) AS failure_rate,
        AVG(c.user_rating) AS avg_rating,
        SUM(CASE WHEN c.is_advanced_feature THEN 1 ELSE 0 END) AS advanced_conversations
    FROM users u
    LEFT JOIN events e ON u.user_id = e.user_id
    LEFT JOIN conversations c ON u.user_id = c.user_id
    GROUP BY 1,2,3,4,5,6,7,8
)
SELECT
    *,
    advanced_conversations / NULLIF(conversations, 0) AS advanced_feature_share
FROM user_activity;
