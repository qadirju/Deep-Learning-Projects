# 📊 Telco Customer Churn ML Pipeline

A comprehensive machine learning pipeline designed to predict customer churn in a telecommunications company. This project includes data preprocessing, model training with hyperparameter tuning, and an interactive Streamlit web application for real-time predictions.

## 🎯 Project Overview

**Objective:** Build a predictive model to identify customers likely to churn (cancel their service) so the company can take proactive retention measures.

**Problem Statement:** Customer churn is a critical business challenge for telecom companies. Predicting which customers are at risk of leaving allows the company to implement targeted retention strategies, ultimately reducing revenue loss and improving customer lifetime value.

**Dataset:** Telco Customer Churn dataset containing approximately 7,000+ customer records with 20+ features including:
- Customer demographics (age, gender, location)
- Service subscriptions (internet, phone, streaming services)
- Account information (tenure, contract type, monthly charges)
- Churn status (target variable)

## ✨ Key Features

- **Automated Data Preprocessing Pipeline**: Handles missing values, feature encoding, and scaling
- **Multiple ML Models**: Logistic Regression, Random Forest with GridSearchCV optimization
- **Interactive Web App**: Streamlit-based UI for single and batch predictions
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Production-Ready**: Serialized models for deployment

## 📁 Project Structure

```
telco-customer-churn-ml-pipeline/
├── README.md                                    # This file
├── LICENSE                                      # Project license
├── requirements.txt                             # Python dependencies
├── STREAMLIT_GUIDE.md                          # Web app setup guide
├── run_streamlit.py                            # Application launcher script
│
├── app/
│   └── streamlit_app.py                        # Streamlit web application
│
├── data/
│   ├── raw/
│   │   └── telco_churn_dataset.csv            # Original dataset
│   └── processed/                              # Preprocessed data (generated)
│
├── models/
│   └── churn_pipeline.joblib                   # Trained model (generated)
│
├── notebooks/
│   └── 01_churn_pipeline_training.ipynb       # Training & analysis notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py                   # Data cleaning & preparation
│   ├── train_model.py                          # Model creation & training
│   └── predict.py                              # Inference module
│
└── reports/
    └── model_evaluation.txt                    # Model performance metrics
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Installation

1. **Clone the repository** (if applicable)
```bash
cd telco-customer-churn-ml-pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Web Application

#### Option 1: Using the Launcher Script (Recommended)
```bash
python run_streamlit.py
```

#### Option 2: Direct Streamlit Command
```bash
streamlit run app/streamlit_app.py
```

The app will open automatically at `http://localhost:8501`

### Training the Model

1. **Open the Jupyter notebook:**
```bash
jupyter notebook notebooks/01_churn_pipeline_training.ipynb
```

2. **Execute all cells** to:
   - Load and explore the data
   - Preprocess features
   - Train and tune the model
   - Evaluate performance
   - Save the model

**Note:** Training includes GridSearchCV which may take 30-60 minutes depending on your hardware.

## 📚 Usage

### Web Application Tabs

#### 1. **Single Prediction**
- Input individual customer information through an interactive form
- Get instant churn prediction with probability score
- Receive recommended retention actions based on the prediction

#### 2. **Batch Prediction**
- Upload a CSV file containing multiple customer records
- Process all customers at once
- Download results as a CSV file for further analysis

#### 3. **About**
- View model information and architecture
- Understanding the methodology and feature importance
- Review key performance metrics and model accuracy

## 🔧 Technical Details

### Data Preprocessing Pipeline

```python
1. Data Loading & Cleaning
   ├── Load raw CSV dataset
   ├── Handle missing values
   └── Convert data types

2. Feature Engineering
   ├── Encode categorical variables (One-Hot Encoding)
   ├── Handle numerical features
   └── Feature scaling (StandardScaler)

3. Train-Test Split
   └── 80-20 split stratified by target variable
```

### Model Training Pipeline

**Models Used:**
- **Logistic Regression**: Fast baseline model for binary classification
- **Random Forest Classifier**: Ensemble method for improved accuracy

**Hyperparameter Tuning:**
- GridSearchCV for systematic parameter optimization
- Cross-validation for robust evaluation

**Performance Metrics:**
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### Feature Importance

The model considers approximately 20+ features including:
- **Demographics**: Senior Citizen, Gender
- **Services**: Internet Service, Phone Service, Streaming Services
- **Account**: Contract Type, Tenure, Monthly Charges, Total Charges
- **Support**: Tech Support, Online Security, Device Protection

## 📊 Model Evaluation

The trained model evaluation results are stored in:
```
reports/model_evaluation.txt
```

Typical performance metrics:
- Accuracy: ~80%+
- ROC-AUC: ~0.85+
- Precision: ~65%+
- Recall: ~55%+

*(Actual metrics depend on model version and training parameters)*

## 🐛 Troubleshooting

### "Model not found" Error
**Solution:** Train the model first by running the Jupyter notebook:
```bash
jupyter notebook notebooks/01_churn_pipeline_training.ipynb
```

### Port 8501 Already in Use
**Solution:** Use a different port:
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Dependencies Missing
**Solution:** Reinstall all requirements:
```bash
pip install -r requirements.txt --upgrade
```

### Streamlit Cache Issues
**Solution:** Clear the cache:
```bash
streamlit cache clear
```

## 📖 Documentation

- **[Streamlit Guide](STREAMLIT_GUIDE.md)** - Detailed web app setup and usage
- **[Training Notebook](notebooks/01_churn_pipeline_training.ipynb)** - In-depth analysis and model development
- **[Source Code](src/)** - Modular Python modules for data processing and predictions

## 🔑 Key Technologies

| Technology | Purpose |
|-----------|---------|
| **scikit-learn** | Machine learning models and preprocessing |
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computations |
| **joblib** | Model serialization |
| **Streamlit** | Interactive web application |
| **Jupyter** | Development and exploration |

## 📈 Future Enhancements

- [ ] Add Deep Learning models (Neural Networks)
- [ ] Implement SHAP for model interpretability
- [ ] Add time-series analysis
- [ ] Deploy to cloud platform (AWS, GCP, Azure)
- [ ] Add customer segmentation analysis
- [ ] Implement automated retraining pipeline

## 📝 Project Flow

```
Raw Data
    ↓
[Data Preprocessing] → Cleaned Data
    ↓
[Train-Test Split]
    ├── Training Set (80%)
    └── Testing Set (20%)
    ↓
[Model Training & Tuning]
    ├── Logistic Regression
    └── Random Forest
    ↓
[Model Evaluation]
    ├── Metrics
    └── Visualizations
    ↓
[Serialized Model] → Streamlit App → User Predictions
```

## 👨‍💻 Development

### Code Structure

- **data_preprocessing.py**: Data loading, cleaning, and feature engineering
- **train_model.py**: Model creation, training, and hyperparameter tuning
- **predict.py**: Inference and predictions on new data
- **streamlit_app.py**: Interactive web interface
- **01_churn_pipeline_training.ipynb**: Comprehensive notebook with EDA and training

### Running Tests

To validate the pipeline with sample predictions:
```bash
python src/predict.py
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Create a new branch (`git checkout -b feature/improvement`)
2. Make your changes
3. Commit with clear messages (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📧 Contact & Support

For questions or support regarding this project:
- Review the [Streamlit Guide](STREAMLIT_GUIDE.md)
- Check the [Jupyter Notebook](notebooks/01_churn_pipeline_training.ipynb) for detailed explanations
- Review source code documentation in the `src/` directory

## 🎓 Learning Outcomes

By exploring this project, you'll learn:
- Building end-to-end ML pipelines
- Data preprocessing and feature engineering techniques
- Model training and hyperparameter tuning
- Model evaluation and metrics interpretation
- Creating production-ready web applications with Streamlit
- Best practices for project organization

---

**Last Updated:** February 2026  
**Status:** ✅ Active & Maintained
