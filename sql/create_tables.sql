CREATE OR REPLACE TABLE users AS
SELECT * FROM read_csv_auto('data/raw/users.csv', header = true);

CREATE OR REPLACE TABLE events AS
SELECT * FROM read_csv_auto('data/raw/events.csv', header = true);

CREATE OR REPLACE TABLE conversations AS
SELECT * FROM read_csv_auto('data/raw/conversations.csv', header = true);

CREATE OR REPLACE TABLE subscriptions AS
SELECT * FROM read_csv_auto('data/raw/subscriptions.csv', header = true);

CREATE OR REPLACE TABLE feedback AS
SELECT * FROM read_csv_auto('data/raw/feedback.csv', header = true);

CREATE OR REPLACE TABLE experiments AS
SELECT * FROM read_csv_auto('data/raw/experiments.csv', header = true);

CREATE OR REPLACE TABLE experiment_assignments AS
SELECT * FROM read_csv_auto('data/raw/experiment_assignments.csv', header = true);
