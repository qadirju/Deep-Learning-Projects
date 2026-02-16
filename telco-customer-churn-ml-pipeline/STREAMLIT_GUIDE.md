# 🚀 Streamlit App - Quick Start Guide

## Running the Telco Churn Predictor Web App

### Method 1: Using the Launcher Script (Recommended)
```bash
cd telco-customer-churn-ml-pipeline
python run_streamlit.py
```

The launcher will:
- ✅ Automatically install Streamlit if needed
- ✅ Launch the web app
- ✅ Open it in your browser at http://localhost:8501

### Method 2: Direct Command
```bash
cd telco-customer-churn-ml-pipeline
streamlit run app/streamlit_app.py
```

### Method 3: Using Python Module
```bash
cd telco-customer-churn-ml-pipeline
python -m streamlit run app/streamlit_app.py
```

## 🌐 Access the App

Once running, open your browser and visit:
```
http://localhost:8501
```

## ✨ Features

The app has three main tabs:

### 1. **Single Prediction** 
   - Input customer information in the form
   - Get instant churn prediction with probability score
   - See recommended retention actions

### 2. **Batch Prediction**
   - Upload a CSV file with multiple customers
   - Get predictions for all customers
   - Download results as CSV

### 3. **About**
   - Model information
   - Methodology and features used
   - Key metrics and performance

## ⚠️ If Model Not Found

The first time you run the app, it will look for the trained model at:
```
models/churn_pipeline.joblib
```

If not found:
1. **Run the training notebook first:**
   ```
   notebooks/01_churn_pipeline_training.ipynb
   ```
2. **Execute cells to train the model** (cells 31-50 depend on GridSearchCV which takes 30-60 minutes)
3. **Save the model** using Cell 47
4. **Restart the Streamlit app**

## 🎯 Keyboard Shortcuts

- **Ctrl+C** - Stop the Streamlit server
- **r** - Rerun the app (in browser)
- **c** - Clear cache

## 📊 Current Model Status

- ✅ Demo pipeline available for testing
- ⏳ Full trained model must be generated from notebook
- 📁 Model location: `models/churn_pipeline.joblib`

## 🔧 Troubleshooting

### "Module not found" Error
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### Port 8501 Already in Use
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Clear Streamlit Cache
```bash
streamlit cache clear
```

---

**Happy predicting! 🎉**
