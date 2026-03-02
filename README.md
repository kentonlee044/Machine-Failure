# 🔧 Predicting Machine Failure Using Machine Learning  
**ENGG2112 – Semester 1, 2025**

A supervised learning project exploring predictive maintenance models across multiple industrial datasets to anticipate machine failure and classify fault types.

---

## 📌 Overview

Unexpected machinery failure leads to:

- Increased maintenance costs  
- Operational downtime  
- Workplace safety risks  

This project investigates machine learning approaches for **predictive maintenance**, shifting from reactive inspection-based systems to data-driven fault detection.

We evaluate multiple supervised learning models across three industrial datasets to:

- Identify the most predictive sensor features  
- Assess generalisation performance  
- Compare binary vs multi-class fault detection  

---

## 📂 Datasets

- Dataset 1 – Sensor-Based Machine Failure Prediction  
  https://www.kaggle.com/datasets/umerrtx/machine-failure-prediction-using-sensor-data  

- Dataset 2 – Predictive Maintenance Classification  
  https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification  

- Dataset 3 – Industrial Fault Detection (Multi-Class)  
  https://www.kaggle.com/datasets/programmer3/industrial-fault-detection-dataset  

---

## 🧠 Methodology

Models evaluated:

- Random Forest  
- Gradient Boosted Trees  
- Randomised Search Cross-Validation  

Analysis techniques:

- Feature Importance Ranking  
- SHAP (SHapley Additive exPlanations)  
- Learning Curve Analysis  

The objective was to build interpretable and generalisable predictive maintenance models.

---

# 📊 Results

## ✅ Dataset 1 – Binary Failure Classification (Random Forest)

**Top Predictive Features**
- VOC (Volatile Organic Compounds)  
- Air Quality (AQ)  
- Ultrasonic Sensor Signals (USS)  

These features capture early operational stress and anomaly signals.

**Insight:**  
The tuned model demonstrated strong generalisation with tightly converging training and validation curves.

---

## ✅ Dataset 2 – Binary Failure Classification (Gradient Boosting)

**Top Predictive Features**
- Tool Wear  
- Torque  
- Rotational Speed  

These directly reflect mechanical stress and degradation.

**Performance**
- Training Score: 0.9976 ± 0.0007  
- Validation Score: 0.9923 ± 0.0040  
- Minimal overfitting  

**Insight:**  
Mechanically meaningful features resulted in highly stable predictive performance.

---

## ⚠️ Dataset 3 – Multi-Class Fault Classification

Classes:
- Normal Operation  
- Overheating  
- Leakage  
- Power Fluctuation  

Included FFT-based vibration features for frequency-domain analysis.

**Key Findings**
- Vibration frequencies most important for normal detection  
- Pressure strongest predictor of overheating  
- Current predictive for leakage  
- Voltage and vibration important for power fluctuations  

**Performance**
- Training Score: 0.8350  
- Validation Score: 0.7159  
- Noticeable overfitting  

**Insight:**  
Multi-class classification proved significantly more challenging, highlighting the impact of data imbalance and feature complexity.

---

# 🧩 Key Takeaways

- Supervised learning performs strongly for **binary machine failure prediction**.
- Model success depends heavily on:
  - Feature engineering  
  - Data quality  
  - Mechanical relevance of inputs  
- Multi-class fault detection requires more balanced and higher-quality data.
- Real-world deployment would require validation on local industrial systems.

---

# 🚀 Future Work

- Collect real-world industrial datasets  
- Improve class balance in multi-class data  
- Reduce correlated FFT features  
- Explore ensemble stacking  
- Develop a lightweight deployment prototype  

---

## 📎 Full Technical Report

Check report/ENGG2112_Final_Report.pdf for the full report or src/ML_Model.ipynb for source code

---
## ⚙️ Installation & Running 

Clone the repository:

```bash
git clone <repo-url>
cd <repo-name>

# install necessary libraries
pip intall -r requirements.txt

# run chunks starting from the top
