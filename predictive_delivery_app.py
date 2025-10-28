import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib


model = joblib.load('delay_predictor_model.pkl')

st.set_page_config(page_title="Predictive Delivery Optimizer", layout="wide")

st.title("🚚 Predictive Delivery Optimizer")
st.markdown("### AI-powered dashboard to predict delivery delays and analyze logistics performance")


@st.cache_data
def load_data():
    orders = pd.read_csv('data/orders.csv')
    delivery = pd.read_csv('data/delivery_performance.csv')
    routes = pd.read_csv('data/routes_distance.csv')
    costs = pd.read_csv('data/cost_breakdown.csv')

    
    for df in [orders, delivery, routes, costs]:
        df.columns = df.columns.str.strip().str.lower()

    
    data = (
        delivery
        .merge(orders, on='order_id', how='left')
        .merge(routes, on='order_id', how='left')
        .merge(costs, on='order_id', how='left')
    )

    
    data['delay_gap_days'] = data['actual_delivery_days'] - data['promised_delivery_days']
    data['total_cost'] = data[['fuel_cost', 'labor_cost', 'vehicle_maintenance',
                               'packaging_cost', 'technology_platform_fee',
                               'other_overhead']].sum(axis=1)
    data['cost_per_km'] = (data['total_cost'] / (data['distance_km'] + 1)).round(2)
    data.fillna(0, inplace=True)

    return data

data = load_data()


st.sidebar.header("🔎 Filter Options")
priority_filter = st.sidebar.multiselect(
    "Select Priority",
    options=data['priority'].dropna().unique(),
    default=data['priority'].dropna().unique()
)

data_filtered = data[data['priority'].isin(priority_filter)]

st.subheader("📊 Delivery Performance Overview")

col1, col2 = st.columns(2)

with col1:
    if 'delivery_status' in data_filtered.columns:
        fig1 = px.pie(
            data_filtered,
            names='delivery_status',
            title='Delivery Status Distribution'
        )
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    if 'customer_rating' in data_filtered.columns:
        fig2 = px.histogram(
            data_filtered,
            x='customer_rating',
            nbins=5,
            title='Customer Rating Distribution'
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

st.subheader("🧭 Distance vs Delivery Time")

if 'distance_km' in data_filtered.columns and 'actual_delivery_days' in data_filtered.columns:
    fig3 = px.scatter(
        data_filtered,
        x='distance_km',
        y='actual_delivery_days',
        color='delivery_status',
        hover_data=['order_id'],
        title='Distance vs Actual Delivery Days'
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")


st.header("🧮 Predict Delivery Delay")

st.write("Enter logistics parameters below to estimate delivery delay probability:")

col1, col2, col3 = st.columns(3)

promised_days = col1.number_input("Promised Delivery Days", min_value=1, value=2)
actual_days = col2.number_input("Actual Delivery Days (so far)", min_value=1, value=2)
distance = col3.number_input("Distance (KM)", min_value=1.0, value=100.0)

fuel = col1.number_input("Fuel Consumption (L)", min_value=1.0, value=20.0)
traffic = col2.number_input("Traffic Delay (Minutes)", min_value=0.0, value=30.0)
order_value = col3.number_input("Order Value (INR)", min_value=100.0, value=5000.0)

fuel_cost = col1.number_input("Fuel Cost (INR)", min_value=0.0, value=1000.0)
labor_cost = col2.number_input("Labor Cost (INR)", min_value=0.0, value=500.0)
packaging_cost = col3.number_input("Packaging Cost (INR)", min_value=0.0, value=200.0)

tech_fee = col1.number_input("Tech Platform Fee (INR)", min_value=0.0, value=100.0)
other_cost = col2.number_input("Other Overhead (INR)", min_value=0.0, value=150.0)
toll = col3.number_input("Toll Charges (INR)", min_value=0.0, value=50.0)


delay_gap = actual_days - promised_days
total_cost = fuel_cost + labor_cost + packaging_cost + tech_fee + other_cost + toll
cost_per_km = total_cost / (distance + 1)


if st.button("🔍 Predict Delay"):
    input_data = pd.DataFrame([[
        promised_days,
        actual_days,
        distance,
        fuel,
        toll,
        traffic,
        order_value,
        fuel_cost,
        labor_cost,
        packaging_cost,
        tech_fee,
        other_cost,
        delay_gap,
        cost_per_km,
        total_cost
    ]], columns=[
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
    ])

    
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    prob = model.predict_proba(input_data)[0][1]
    st.success(f"Predicted Delay Probability: {prob*100:.2f}%")

    if prob > 0.5:
        st.warning("⚠️ High chance of delay! Consider adjusting routes or assigning faster carriers.")
    else:
        st.info("✅ Delivery likely on time!")

st.markdown("---")
st.caption("Developed by Bhavya | OFI Services AI Internship Case Study 2025")
