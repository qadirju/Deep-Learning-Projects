# 🖼️ CNN Scene Classification

This repository contains a **Convolutional Neural Network (CNN)** implementation for **scene classification** into six categories:  
`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

The project demonstrates data loading, preprocessing, CNN model design, training, validation, testing, augmentation, and single image prediction using **TensorFlow / Keras**.

---

## 📂 Dataset

- **Assignment dataset (small version):** [Google Drive Link](https://drive.google.com/drive/folders/1bevGzEJ7pXbUOWYmRFs6pImmEd6zarcH?usp=sharing)  
- **Original full dataset:** [Kaggle - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/)

The dataset consists of:
- **Train set:** 100 images  
- **Validation set:** 44 images (from train folder split)  
- **Test set:** 100 images  

---

## 🛠️ Tasks Completed

1. **Dataset Preparation**  
   - Loaded train/validation/test sets using `image_dataset_from_directory`  
   - Used `label_mode='categorical'` (since this is a multiclass classification task)

2. **CNN Model**  
   - Built a CNN with feature maps: **32 → 64 → 128 → 256 → 512 → 1024**  
   - Each block: `Conv2D(kernel_size=3, activation='relu')` + `MaxPooling2D(pool_size=2)`  
   - Flatten → Dense → Softmax (6 outputs)

3. **Model Compilation**  
   - Loss: `categorical_crossentropy`  
   - Optimizer: `Adam`  
   - Metric: `accuracy`

4. **Training**  
   - Trained for **20 epochs**  
   - Validated on 44 images  
   - Plotted training & validation accuracy to check for overfitting

5. **Testing**  
   - Evaluated on **100 test images**

6. **Model with Augmentation**  
   - Added augmentation layer (`RandomFlip(vertical)`, `RandomRotation(0.3)`, `RandomZoom(0.3)`)  
   - Added Dropout(0.5) for regularization  
   - Retrained and compared with baseline model

7. **Performance Comparison**  
   - Compared baseline vs augmented model performance on validation and test sets  
   - Wrote analysis on generalization and overfitting

8. **Prediction Function**  
   - Implemented function to predict class of any custom image path

---

## 📊 Model Architecture (Baseline)

