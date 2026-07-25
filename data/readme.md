# Data: customer churn dataset

## Context

This dataset simulates 6,000 customers of a fintech remittance app. Each
row is one customer, described by their tenure, transaction behavior,
support interaction history and whether they churned.

The data is entirely synthetic (generated with numpy, seed fixed for
reproducibility). Churn probability was engineered as a function of the
features below, with realistic noise added, so the relationships a model
finds are the same kind of relationships you'd expect to find in a real
churn dataset: recency and support friction matter a lot, tenure and
engagement are protective.

## File: customer_churn_data.csv

One row per customer.

| Column | Type | Description |
|---|---|---|
| `customer_id` | integer | Unique identifier for the customer |
| `tenure_months` | integer | How many months the customer has been active |
| `avg_monthly_transactions` | float | Average number of transfers per month |
| `avg_transaction_amount_usd` | float | Average transfer amount, in USD |
| `days_since_last_transaction` | integer | Days since the customer's last transfer, as of the reference date |
| `support_tickets_last_90d` | integer | Number of support tickets raised in the last 90 days |
| `avg_satisfaction_score` | float (0-10) | Average post-support satisfaction score. Missing if the customer never contacted support |
| `used_promo_code` | integer (0/1) | Whether the customer has used a promotional code |
| `referred_by_friend` | integer (0/1) | Whether the customer was referred by another user |
| `country` | text | Customer's country |
| `device` | text | `Android` or `iOS` |
| `churned` | integer (0/1) | Target variable: 1 if the customer churned (no transfer in the last 60 days), 0 otherwise |

## Why `avg_satisfaction_score` has missing values

It's only populated for customers who contacted support at least once
(`support_tickets_last_90d > 0`). A customer who never contacted support
has no satisfaction survey to report. This is handled explicitly in the
notebook's feature engineering step, by creating an `ever_contacted_support`
flag before filling the missing values, rather than silently imputing a
mean and losing that signal.

## How the data was generated

Churn probability was modeled as a logistic function of the features
above:

- **Increases churn risk:** more days since the last transaction, more
  support tickets in the last 90 days
- **Decreases churn risk:** longer tenure, higher average monthly
  transactions, higher satisfaction score, being referred by a friend

Random noise was added on top of this signal, and the resulting
probability was used to simulate the binary `churned` outcome. The overall
churn rate in the dataset is 17.8%.
