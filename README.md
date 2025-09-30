
# AeroVerse: Extended Reality & AI-Based Airport Navigation System

## 📌 Overview
AeroVerse is a graduation project that integrates **Artificial Intelligence (AI)** and **Extended Reality (XR)** to enhance airport safety, particularly during the **critical landing phase**.  
The system combines **real-time weather forecasting**, **weather image classification**, and **runway object detection** with XR visualization to support **pilots, air traffic controllers (ATC), and ground staff** in making faster and safer decisions.

---

## 🚩 Problem Statement
- 47–53% of aviation accidents occur during landing.
- Adverse weather (fog, rain, strong winds) increases risks of low-visibility operations.
- Runway congestion and obstacles (birds, vehicles, FOD) lead to delays, fuel waste, and higher operational costs.

---

## 🛠️ Methodology
### 🔹 AI Modules
- **Weather Forecasting** → XGBoost model trained on historical & real-time weather data.  
- **Weather Image Classification** → EfficientNetB0 CNN to classify in-flight images (rain, fog, snow, clear sky, etc.).  
- **Runway Object Detection** → YOLO-based model to detect hazards (aircraft, vehicles, birds, obstacles).  

### 🔹 XR Integration
- 3D airport assets designed using **Blender & 3ds Max**.  
- Immersive simulation built in **Unity** for **Meta Quest 3**.  
- Provides blind-zone visualization and runway alerts in real-time.  

### 🔹 System Deployment
- **FastAPI** for serving AI models.  
- **Edge computing** to ensure real-time, low-latency responses.  
- **Centralized dashboard** integrates predictions, detections, and XR environment.

---

## 📊 Results
- ✅ Accurate 24-hour weather forecasts.  
- ✅ Real-time classification of sky conditions from cockpit images.  
- ✅ Runway object detection improves landing/takeoff safety.  
- ✅ XR-based visualization provides immersive training and operational awareness.  

---

## 🌍 Impact
- **Enhanced aviation safety** during landing operations.  
- **Reduced delays & fuel consumption** through better decision-making.  
- **Improved pilot & ATC training** using XR simulation.  
- **Paving the way** for smarter and safer airports worldwide.  

---
