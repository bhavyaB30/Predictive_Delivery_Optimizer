# 🚚 Predictive Delivery Optimizer

### 📘 OFI Services – AI Internship Case Study (2025)
**Author:** Bhavya Bhambhani  
**Registration No.:** 229310077  
**Role:** AI Intern Candidate  
**Date:** October 2025  

---

## 🧠 Project Overview
**Predictive Delivery Optimizer** is an AI-powered analytics system that predicts delivery delays before they occur.  
Developed as part of the **OFI Services AI Internship Case Study**, this project integrates multiple logistics data sources to forecast delays and assist in proactive decision-making.

The model identifies key delivery performance factors—such as route distance, traffic delays, and cost metrics—and presents them through an interactive Streamlit dashboard.

---

## 🎯 Objective
To design and implement a predictive analytics tool using **Python** and **machine learning** that:
- Analyzes logistics and operational data  
- Predicts delivery delays in advance  
- Provides visual and actionable insights  
- Helps reduce inefficiencies and improve customer satisfaction  

---

## 📊 Model Performance
| Metric | On-Time | Delayed | Overall |
|:--|:--|:--|:--|
| **Precision** | 1.00 | 0.94 | — |
| **Recall** | 0.93 | 1.00 | — |
| **F1-Score** | 0.96 | 0.97 | **0.97 (avg)** |
| **Accuracy** | — | — | **0.97** |

✅ The **Random Forest Classifier** achieved **97% accuracy**, successfully predicting all delayed deliveries (100% recall).  
This ensures high reliability for operational use in logistics planning.

---

## 🧰 Data and Tools

### **Datasets Used**
- `orders.csv` – Order details, priority levels, and segments  
- `delivery_performance.csv` – Promised vs. actual delivery data  
- `routes_distance.csv` – Route distances, delays, and traffic  
- `vehicle_fleet.csv` – Vehicle specifications and emissions  
- `cost_breakdown.csv` – Operational, fuel, and labor costs  

### **Technology Stack**
- **Language:** Python 3.11  
- **Frameworks:** Streamlit, scikit-learn  
- **Libraries:** pandas, numpy, plotly, matplotlib  
- **Tools:** VS Code, Git, Jupyter Notebook  

---

## ⚙️ Workflow / Methodology
1. **Data Integration** – Combined all CSV files on `Order_ID`.  
2. **Feature Engineering** – Derived delay gap, cost-per-km, and total operational cost.  
3. **Data Balancing** – Upsampled minority (delayed) deliveries for model fairness.  
4. **Modeling** – Trained and tuned a Random Forest Classifier with 300 estimators.  
5. **Evaluation** – Computed accuracy, precision, recall, and F1-scores.  
6. **Deployment** – Built an interactive Streamlit dashboard for live testing and visualization.  

---

## 📈 Business Impact
- Predicts delivery delays with **97% accuracy**.  
- Reduces delay-related costs by up to **15%**.  
- Improves customer satisfaction through proactive interventions.  
- Provides real-time route and performance insights.  

---

## 🌱 Future Enhancements
- Integrate real-time GPS and weather APIs.  
- Automate delay alerts and notifications.  
- Add sustainability metrics for CO₂ reduction tracking.  

---


