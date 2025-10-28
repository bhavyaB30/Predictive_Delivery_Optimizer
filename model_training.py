import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
import joblib
import numpy as np


orders = pd.read_csv('data/orders.csv')
delivery = pd.read_csv('data/delivery_performance.csv')
routes = pd.read_csv('data/routes_distance.csv')
vehicles = pd.read_csv('data/vehicle_fleet.csv')
costs = pd.read_csv('data/cost_breakdown.csv')


for df in [orders, delivery, routes, vehicles, costs]:
    df.columns = df.columns.str.strip().str.lower()


df = (
    delivery
    .merge(orders, on='order_id', how='left')
    .merge(routes, on='order_id', how='left')
    .merge(costs, on='order_id', how='left')
)


df['delay'] = (df['actual_delivery_days'] > df['promised_delivery_days']).astype(int)
df['delay_gap_days'] = df['actual_delivery_days'] - df['promised_delivery_days']

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)


df['cost_per_km'] = (
    (df['fuel_cost'] + df['labor_cost'] + df['toll_charges_inr'] + df['packaging_cost']) /
    (df['distance_km'] + 1)
)
df['total_cost'] = df[['fuel_cost', 'labor_cost', 'vehicle_maintenance',
                       'packaging_cost', 'technology_platform_fee',
                       'other_overhead']].sum(axis=1)


features = [
    'promised_delivery_days',
    'actual_delivery_days',
    'distance_km',
    'fuel_consumption_l',
    'toll_charges_inr',
    'traffic_delay_minutes',
    'order_value_inr',
    'fuel_cost',
    'labor_cost',
    'packaging_cost',
    'technology_platform_fee',
    'other_overhead',
    'delay_gap_days',
    'cost_per_km',
    'total_cost'
]

features = [f for f in features if f in df.columns]

X = df[features]
y = df['delay']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


train_df = pd.concat([X_train, y_train], axis=1)
majority = train_df[train_df.delay == 0]
minority = train_df[train_df.delay == 1]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

train_balanced = pd.concat([majority, minority_upsampled])
X_train = train_balanced.drop('delay', axis=1)
y_train = train_balanced['delay']

print(f"✅ After balancing: {y_train.value_counts().to_dict()}")


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=4,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("\n✅ Model trained successfully!")
print(f"📊 Accuracy: {acc:.2f}")
print(f"🎯 F1 Score: {f1:.2f}")
print(f"🔥 ROC-AUC: {roc_auc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


joblib.dump(model, 'delay_predictor_model.pkl')
print("\n💾 Model saved as delay_predictor_model.pkl")
