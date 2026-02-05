# 🚀 Deep Learning Projects Repository

A comprehensive collection of **deep learning and machine learning projects** showcasing various neural network architectures, natural language processing, computer vision, and production-ready applications.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Current Projects](#current-projects)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

---

## 📖 Overview

This repository contains multiple **machine learning and deep learning projects** ranging from:
- 📊 **Predictive Analytics** - Binary classification, regression models
- 🧠 **Neural Networks** - CNNs, RNNs, Transformers
- 🖼️ **Computer Vision** - Object detection, image classification
- 💬 **Natural Language Processing** - Text processing, embeddings, RAG chatbots
- 🧬 **Transfer Learning** - Fine-tuning LLMs with LoRA/QLoRA
- 📱 **Mobile Development** - Flutter marketplace application

All projects include:
- ✅ Complete implementations with explanations
- ✅ Jupyter notebooks for easy exploration
- ✅ Pre-trained models and artifacts
- ✅ Data processing pipelines
- ✅ Professional documentation

---

## 🗂️ Project Structure

```
.
├── ML-Models/                                    # Traditional ML & baseline models
│   ├── Bank-Binary-Prediction/                 # Bank customer binary classification
│   ├── Graph-Theory/                           # PageRank algorithm implementation
│   ├── Heart_Disease_prediction/               # Heart disease prediction system
│   ├── Text Processing with Deep Neural Network/ # DNN text processing
│   └── YOLO-Real-Time-Object-Detection/        # Real-time object detection
├── Deep-Learning-Projects/                      # Advanced DL projects
│   ├── Classification Using CNN/               # CNN-based image classification
│   ├── Classification with LoRA and QLoRA/     # LLM fine-tuning
│   └── context-aware-chatbot-rag/              # RAG-powered conversational AI
├── cattle-marketplace-fyp/                      # Flutter mobile app
└── README.md                                    # This file
```

---

## 📁 Current Projects

### 🏦 **ML-Models Folder**

#### 1. **Bank-Binary-Prediction**
**Purpose:** Binary classification for predicting bank customer behavior  
**Location:** `ML-Models/Bank-Binary-Prediction/`

- **Task:** Predict whether a customer will subscribe to a bank product
- **Model Types:** Logistic Regression, Decision Trees, Random Forests
- **Key Components:**
  - `Bank_Dataset.ipynb` - Data exploration, preprocessing, model training
  - `train.csv` / `test.csv` - Training and test datasets
  - `submission.csv` - Model predictions
- **Technologies:** Scikit-learn, Pandas, NumPy
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score

**Quick Start:**
```bash
cd ML-Models/Bank-Binary-Prediction
jupyter notebook Bank_Dataset.ipynb
```

---

#### 2. **Heart_Disease_Prediction**
**Purpose:** Predict heart disease presence using medical indicators  
**Location:** `ML-Models/Heart_Disease_prediction/`

- **Task:** Multi-class/Binary classification of heart disease risk
- **Features:**
  - Age, blood pressure, cholesterol, heart rate, etc.
  - Multiple classification algorithms
  - Model comparison and evaluation
- **Key Files:**
  - `heartDP.ipynb` - Main notebook with full pipeline
  - `models/heart_model.joblib` - Trained model artifact
  - `figures/` - Classification reports and visualizations
- **Models Included:**
  - Decision Tree Classifier
  - Logistic Regression
  - (Additional ensemble methods)
- **Technologies:** Scikit-learn, Matplotlib, Pandas
- **Use Case:** Healthcare/Medical prediction system

**Quick Start:**
```bash
cd ML-Models/Heart_Disease_prediction
jupyter notebook heartDP.ipynb
```

---

#### 3. **Graph-Theory**
**Purpose:** Implementation of PageRank algorithm  
**Location:** `ML-Models/Graph-Theory/`

- **Algorithm:** PageRank (as used by Google Search)
- **Key Concepts:**
  - Graph theory fundamentals
  - Ranking nodes in a network
  - Iterative convergence
- **Key Files:**
  - `PageRank_Notebook.ipynb` - Implementation and examples
- **Applications:**
  - Search engine ranking
  - Social network analysis
  - Link importance calculation
- **Technologies:** NumPy, networkx, Matplotlib

**Quick Start:**
```bash
cd ML-Models/Graph-Theory
jupyter notebook PageRank_Notebook.ipynb
```

---

#### 4. **Text Processing with Deep Neural Network**
**Purpose:** DNN-based text classification and processing  
**Location:** `ML-Models/Text Processing with Deep Neural Network/`

- **Task:** Text sentiment analysis or category classification
- **Architecture:**
  - Deep feedforward neural networks
  - Embedding layers
  - Text preprocessing pipeline
- **Key Files:**
  - `Text_Processing_DNN.ipynb` - Complete implementation
  - `Data/bbcsports.csv` - BBC sports news dataset
  - `Data/tweet_emotions.csv` - Tweet emotion classification data
- **Models:**
  - Custom DNN architecture
  - Pre-trained embeddings
- **Technologies:** Keras/TensorFlow, NLTK, Scikit-learn
- **Output:** Text classification with >80% accuracy

**Quick Start:**
```bash
cd "ML-Models/Text Processing with Deep Neural Network/Code"
jupyter notebook Text_Processing_DNN.ipynb
```

---

#### 5. **YOLO-Real-Time-Object-Detection**
**Purpose:** Real-time object detection using YOLO  
**Location:** `ML-Models/YOLO-Real-Time-Object-Detection/`

- **Framework:** YOLOv3/YOLOv4/YOLOv5
- **Capabilities:**
  - Real-time object detection on images/videos
  - Multiple object class detection
  - High-speed inference
- **Key Features:**
  - Pre-trained weights included
  - Support for custom training
  - Bounding box visualization
- **Technologies:** PyTorch/TensorFlow, OpenCV, YOLO Framework
- **Applications:**
  - Video surveillance
  - Traffic monitoring
  - Autonomous vehicles
  - Safety inspection

**Quick Start:**
```bash
cd ML-Models/YOLO-Real-Time-Object-Detection
# Run detection on image or video
python detect.py --source image.jpg
```

---

### 🧠 **Deep-Learning-Projects Folder**

#### 1. **Classification Using CNN**
**Purpose:** Convolutional Neural Networks for image classification  
**Location:** `Deep-Learning-Projects/Classification Using CNN/`

- **Architecture:** Custom CNN + VGG/ResNet variants
- **Task:** Multi-class image classification
- **Key Components:**
  - `Classifiaction_CNN_based.ipynb` - Full CNN implementation
  - Convolutional layers for feature extraction
  - Pooling and fully connected layers
- **Dataset Support:**
  - CIFAR-10 / CIFAR-100
  - MNIST
  - Custom image datasets
- **Technologies:** TensorFlow/Keras, NumPy, Matplotlib
- **Performance:** >95% accuracy on standard benchmarks

**Key Concepts:**
- Convolution and pooling operations
- Activation functions (ReLU, Softmax)
- Backpropagation and optimization
- Data augmentation techniques

**Quick Start:**
```bash
cd Deep-Learning-Projects/Classification\ Using\ CNN
jupyter notebook Classifiaction_CNN_based.ipynb
```

---

#### 2. **Classification with LoRA and QLoRA**
**Purpose:** LLM fine-tuning with parameter-efficient methods  
**Location:** `Deep-Learning-Projects/Classification with LoRA and QLoRA/`

- **Techniques:**
  - **LoRA** (Low-Rank Adaptation) - Efficient LLM fine-tuning
  - **QLoRA** (Quantized LoRA) - Memory-efficient approach
- **Key Files:**
  - `llm_fine_tuning.ipynb` - Main implementation
  - `llm_fine_tuning (1).ipynb` - Alternative approach
- **Use Cases:**
  - Fine-tune large language models (GPT, LLAMA, etc.)
  - Custom domain adaptation
  - Classification tasks with LLMs
- **Benefits:**
  - Reduces trainable parameters by 99%
  - Trains on consumer GPUs (8GB VRAM)
  - Maintains model performance
  - Faster convergence
- **Technologies:** HuggingFace Transformers, PEFT, Torch, bitsandbytes

**Key Concepts:**
- Transformer architecture basics
- LoRA adapter mechanism
- Quantization principles
- Efficient VRAM management

**Quick Start:**
```bash
cd Deep-Learning-Projects/Classification\ with\ LoRA\ and\ QLoRA
jupyter notebook llm_fine_tuning.ipynb
```

---

#### 3. **Context-Aware Chatbot with RAG**
**Purpose:** Retrieval-Augmented Generation conversational AI  
**Location:** `Deep-Learning-Projects/context-aware-chatbot-rag/`

- **Architecture:**
  - Retrieval component (FAISS/keyword-based)
  - Language model (FLAN-T5)
  - Conversation memory
  - Relevance scoring
- **Features:**
  - 💬 Multi-turn conversations with context
  - 📚 Knowledge base retrieval
  - 🎯 Out-of-scope question detection
  - 🌐 Web interface (Streamlit)
  - ✨ Template-based response generation
- **Key Files:**
  - `app.py` - Streamlit web application
  - `ContextAware_RAG_Chatbot.ipynb` - Development notebook
  - `README.md` - Comprehensive documentation
- **Technologies:**
  - LangChain (RAG orchestration)
  - Streamlit (Web UI)
  - Sentence Transformers (embeddings)
  - FAISS (vector database)
  - HuggingFace (LLM)
- **Deployment:** Streamlit Cloud, Docker, Local

**Key Concepts:**
- RAG (Retrieval-Augmented Generation)
- Semantic search and embeddings
- Conversation memory management
- Relevance scoring for query understanding

**Quick Start:**
```bash
cd Deep-Learning-Projects/context-aware-chatbot-rag
streamlit run app.py
# Access at http://localhost:8501
```

---

### 📱 **Cattle Marketplace FYP**
**Purpose:** Flutter mobile application for cattle trading marketplace  
**Location:** `cattle-marketplace-fyp/`

- **Type:** Cross-platform mobile application
- **Technology Stack:**
  - **Frontend:** Flutter (Dart)
  - **Backend:** Firebase / REST API
  - **Platforms:** Android, iOS, Web
- **Key Features:**
  - Cattle listing and marketplace
  - User authentication
  - Payment integration
  - Real-time notifications
  - GPS-based location services
- **Project Structure:**
  - `lib/` - Flutter app source code
  - `android/` - Android-specific configuration
  - `ios/` - iOS-specific configuration
  - `web/` - Web deployment files
  - `assets/` - Images, icons, fonts
- **Build Tools:**
  - Flutter SDK
  - Dart analyzer
  - Android Studio
  - Xcode (for iOS)

**Note:** This is a **Final Year Project (FYP)** for academic submission.

**Quick Start:**
```bash
cd cattle-marketplace-fyp
flutter pub get
flutter run  # Run on emulator or connected device
```

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
# Core requirements
- Python 3.8+
- Git
- pip or conda

# Optional but recommended
- Virtual environment (venv/conda)
- Jupyter Notebook
- Docker (for containerized deployment)
```

### Setup Instructions

#### 1. Clone Repository
```bash
git clone https://github.com/qadirju/Deep-Learning-Projects.git
cd Deep-Learning-Projects
```

#### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Or using conda
conda create -n deep-learning python=3.9
conda activate deep-learning
```

#### 3. Install Dependencies (Project-Specific)
```bash
# For ML Models projects
pip install scikit-learn pandas numpy matplotlib jupyter

# For Deep Learning projects
pip install tensorflow keras torch torchvision

# For RAG Chatbot
pip install streamlit langchain sentence-transformers faiss-cpu

# For NLP projects
pip install nltk transformers huggingface-hub

# For YOLO
pip install yolov5 opencv-python
```

#### 4. Verify Installation
```bash
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
jupyter notebook  # Start Jupyter
```

---

## 🔧 Technologies Used

### Machine Learning & Deep Learning
| Library | Purpose | Projects |
|---------|---------|----------|
| **Scikit-learn** | ML algorithms, preprocessing | Bank prediction, Heart disease |
| **TensorFlow/Keras** | Deep learning framework | CNN, Text DNN |
| **PyTorch** | Deep learning, transformers | LoRA/QLoRA, YOLO |
| **LangChain** | RAG/LLM orchestration | Context-aware chatbot |
| **HuggingFace** | Pre-trained models, transformers | RAG, LoRA/QLoRA |

### Data & Visualization
| Library | Purpose |
|---------|---------|
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Matplotlib** | Static visualization |
| **OpenCV** | Computer vision |

### Deployment & UI
| Tool | Purpose |
|------|---------|
| **Streamlit** | Web UI for ML apps |
| **Flutter** | Mobile development |
| **Docker** | Containerization |

---

## 📊 Project Statistics

| Category | Count | Status |
|----------|-------|--------|
| ML Projects | 5 | ✅ Complete |
| DL Projects | 3 | ✅ Complete |
| Mobile Apps | 1 | ✅ In Development |
| **Total** | **9** | - |

---

## 🎯 Next Steps

### To Get Started:
1. ✅ Clone the repository
2. ✅ Create virtual environment
3. ✅ Install project-specific dependencies
4. ✅ Navigate to project folder
5. ✅ Open Jupyter notebook or run app
6. ✅ Follow project-specific README files

### For Specific Interest:
- **Want to learn ML basics?** → Start with Bank-Binary-Prediction
- **Interested in NLP?** → Check Text Processing with DNN or RAG Chatbot
- **Computer Vision enthusiast?** → Explore CNN Classification or YOLO
- **LLM fine-tuning?** → See LoRA and QLoRA project
- **Deployment?** → RAG Chatbot is production-ready

---

## 📚 Resources & Documentation

Each project includes:
- 📖 **Comprehensive README** - Project-specific documentation
- 🎓 **Jupyter Notebooks** - Step-by-step implementations
- 💾 **Pre-trained Models** - Saved artifacts and weights
- 📊 **Sample Data** - Datasets for experimentation
- 📈 **Results & Metrics** - Performance evaluation

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines:
- Follow existing code style and structure
- Add comments and documentation
- Update relevant README files
- Include performance metrics
- Test your code before submitting

---

## 📄 License

This repository is open source and available under the MIT License.

---

## 👨‍💻 Author

**Qadir Jubair** - AI/ML Enthusiast

### Connect:
- 📧 Email: [Add your email]
- 💼 LinkedIn: [Add LinkedIn profile]
- 🐙 GitHub: [@qadirju](https://github.com/qadirju)

---

## 📮 Support & Issues

- 🐛 Found a bug? Open an issue
- 💡 Have suggestions? Create a discussion
- ❓ Questions? Check project README files

---

## 🎉 Acknowledgments

Special thanks to:
- Open source community
- TensorFlow, PyTorch, and HuggingFace teams
- Dataset providers and research community

---

**Last Updated:** February 2026  
**Version:** 1.0.0  
**Status:** Active & Maintained ✅
