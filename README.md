# 🫀 Diagnostic Cardiology AI

A sophisticated machine learning application that predicts the likelihood of heart disease based on clinical parameters. Built with a cost-complexity pruned Decision Tree algorithm and wrapped in an elegant, interactive Streamlit interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)

---

## 🎯 Overview

**Diagnostic Cardiology AI** is an intelligent diagnostic tool designed to assist healthcare professionals and individuals in assessing cardiovascular health risk. By analyzing 13 clinical parameters—including vital signs, symptoms, and cardiac imaging results—the system leverages machine learning to estimate the probability of heart disease presence.

This project demonstrates the practical intersection of data science and healthcare, showcasing how machine learning can be deployed as an accessible, user-friendly application for real-world scenarios.

---

## ✨ Features

### 🎨 **Beautiful Aurora-Inspired UI**
- Animated gradient background with aurora effect
- Glassmorphism design for input cards
- Neon glowing title with smooth animations
- Responsive layout optimized for all screen sizes
- Interactive hover effects and smooth transitions

### 🔍 **Comprehensive Clinical Input**
Organized into three intuitive sections:

1. **Vitals & Demographics**
   - Age
   - Sex (Male/Female)
   - Resting Blood Pressure
   - Serum Cholesterol
   - Fasting Blood Sugar Status

2. **Clinical Symptoms**
   - Chest Pain Type (4 classifications)
   - Exercise Induced Angina
   - Maximum Heart Rate Achieved

3. **ECG & Imaging**
   - Resting ECG Results (3 classifications)
   - ST Depression Induced by Exercise
   - Slope of Peak Exercise ST Segment
   - Number of Major Vessels Colored
   - Thalassemia Status

### 📊 **Instant Diagnostic Results**
- Binary classification (High Risk / Low Risk)
- Confidence Score (0-100%)
- Visual progress indicator
- Color-coded result cards (red for risk, green for safety)

### 🚀 **Production-Ready Pipeline**
- Automatic median imputation for missing values
- One-hot encoding for categorical features
- Serialized model pipeline for consistent predictions
- Fast inference with caching

---

## 🛠️ Technology Stack

### **Backend & Core**
- **Python 3.x** — Primary programming language
- **scikit-learn** — Machine learning framework
  - Decision Tree Classifier (cost-complexity pruned)
  - Pipeline for preprocessing and prediction
  - SimpleImputer for handling missing values
  - OneHotEncoder for categorical features

### **Frontend & UI**
- **Streamlit** — Interactive web application framework
  - Rapid deployment with minimal boilerplate
  - Component-based layout system
  - Real-time interactivity
  - Custom CSS injection for advanced styling

### **Data & Serialization**
- **pandas** — Data manipulation and DataFrame handling
- **joblib** — Model serialization and deserialization
  - Efficient binary format for large models
  - Preserves entire pipeline structure

### **Styling & Animations**
- **CSS3** — Advanced styling and animations
  - Keyframe animations for aurora background
  - Glassmorphism effects with backdrop-filter
  - Smooth transitions and hover states
  - Linear gradients and color blending

---

## 📦 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Chandu-gummadavelly/Heart-disease-predictor.git
cd Heart-disease-predictor
