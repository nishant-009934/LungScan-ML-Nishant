# Lung Cancer Detection Using Machine Learning

An AI-powered Lung Cancer Detection system that uses Deep Learning and Medical Image Processing techniques to analyze CT scan images and predict whether lung nodules are Benign or Malignant.

This project aims to assist in the early detection of lung cancer, helping improve diagnosis speed and accuracy through Machine Learning.

---

# Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection can significantly improve survival rates and treatment effectiveness.

This project uses a Convolutional Neural Network (CNN) model trained on lung CT scan images to automatically classify lung nodules. The system performs image preprocessing, feature extraction, model prediction, and displays results through a simple Streamlit web application.

---

# Features

- Lung cancer detection using Deep Learning
- CT scan image preprocessing
- CNN-based classification model
- Benign vs Malignant prediction
- Streamlit web interface
- Medical image visualization
- Model evaluation and accuracy analysis
- User-friendly interface

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| OpenCV | Image preprocessing |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Scikit-learn | Evaluation metrics |
| Matplotlib | Data visualization |
| Streamlit | Web application interface |

---

# Project Structure

```text
LungScan-Pro/
│
├── app/
│   └── app.py
│
├── dataset/
│   ├── benign/
│   └── malignant/
│
├── models_saved/
│   └── lung_cnn_model.pt
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── training/
│
├── README.md
├── requirements.txt
├── .gitignore
└── main.py
```

---

# Dataset

The project uses Lung CT Scan datasets for training and testing the model.

### Classes:
- Benign Lung Nodules
- Malignant Lung Nodules

The dataset images are preprocessed before training to improve model performance and prediction accuracy.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/lung-cancer-detection.git
```

## 2. Navigate to the Project Folder

```bash
cd lung-cancer-detection
```

## 3. Create Virtual Environment (Optional but Recommended)

### Windows

```bash
python -m venv .venv
```

### Activate Environment

```bash
.venv\Scripts\activate
```

## 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the Streamlit application:

```bash
streamlit run app/app.py
```

After running the command, open the local server URL shown in the terminal:

```text
http://localhost:8501
```

---

# Model Workflow

1. Data Collection  
2. Image Preprocessing  
3. Data Augmentation  
4. Feature Extraction  
5. CNN Model Training  
6. Model Evaluation  
7. Prediction and Visualization  

---

# Evaluation Metrics

The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

---

# Application Output

The application predicts:

- Benign Lung Nodule
- Malignant Lung Nodule

along with prediction confidence scores and image visualization.

---

# Future Improvements

- Improve model accuracy using advanced architectures
- Add Explainable AI (XAI) visualization
- Deploy on cloud platforms
- Integrate real-time hospital systems
- Multi-class lung disease classification
- Add support for DICOM image files

---

# Research References

- LIDC-IDRI Dataset
- LUNA16 Dataset
- Medical Imaging Research Papers
- IEEE Research Articles

---

# Author

Developed as a Machine Learning and Medical Imaging project for academic, research, and educational purposes.

---

# License

This project is intended for educational and research purposes only.