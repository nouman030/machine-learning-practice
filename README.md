# 🤖 Machine Learning & AI Engineering Practice

Welcome to my AI Engineering repository! This project serves as a practical log of my journey through the core concepts of Machine Learning. It includes hands-on implementations of various data preprocessing techniques and ML models, built from the ground up.

This repository is organized to follow a standard ML pipeline, from data processing to model implementation. Each folder contains Jupyter notebooks with clear explanations and code.

## 📜 Table of Contents
- [🤖 Machine Learning \& AI Engineering Practice](#-machine-learning--ai-engineering-practice)
  - [📜 Table of Contents](#-table-of-contents)
  - [📂 Repository Structure](#-repository-structure)
  - [📘 Topics Covered](#-topics-covered)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Supervised Learning](#2-supervised-learning)
    - [3. Unsupervised Learning](#3-unsupervised-learning)
  - [🛠 Tech Stack](#-tech-stack)
  - [🚀 Getting Started](#-getting-started)
  - [📊 Example Output](#-example-output)
  - [🔮 Future Plans](#-future-plans)

---

## 📂 Repository Structure

The repository is structured logically to reflect the typical machine learning workflow:

```bash
├── Data_Preprocessing/
│   ├── Encoding.ipynb
│   ├── Scaling.ipynb
│   ├── Outliers.ipynb
│   └── Imputation.ipynb
├── Supervised_Learning/
│   ├── Linear_Regression.ipynb
│   ├── Logistic_Regression.ipynb
│   ├── Decision_Tree.ipynb
│   ├── Random_Forest.ipynb
│   ├── Naive_Bayes.ipynb
│   ├── SVM.ipynb
│   └── KNN.ipynb
├── Unsupervised_Learning/
│   └── KMeans.ipynb
└── utils/
    ├── data_loader.py
    ├── helpers.py
    └── __init__.py
```

---

## 📘 Topics Covered

This repository covers a wide range of fundamental ML concepts and algorithms:

### 1. Data Preprocessing
Essential techniques to clean and prepare data for modeling.
- **Handling Missing Data**: Strategies for imputing missing values using mean, median, and mode.
- **Feature Encoding**: Converting categorical data into a numerical format.
  - Label Encoding
  - Ordinal Encoding
  - One-Hot Encoding
  - Binary Encoding
  - Dummy Variables
- **Feature Scaling**: Normalizing the range of features to improve model performance.
  - Standard Scaler
  - Min–Max Scaler
  - Robust Scaler
  - Max Abs Scaler
- **Outlier Handling**: Techniques for detecting and managing anomalous data points.

### 2. Supervised Learning
Models that learn from labeled data to make predictions.
- **Regression Models (Predicting Continuous Values)**
  - Linear Regression (Simple & Multiple)
  - Polynomial Regression (For non-linear relationships)
  - Ridge & Lasso Regression (L1 & L2): Regularization techniques to prevent overfitting

- **Classification Models (Predicting Categories)**
  - Logistic Regression
  - Naive Bayes (Gaussian, Multinomial, Bernoulli)
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM)

### 3. Unsupervised Learning
Models that find hidden patterns and structures in unlabeled data.
- **K-Means Clustering**: Group data points into K distinct clusters.

---

## 🛠 Tech Stack

This project is built on the standard Python data science and machine learning ecosystem.
- **Language**: Python 3
- **Libraries**:  
  - scikit-learn (For ML models and preprocessing)  
  - pandas (For data manipulation)  
  - numpy (For numerical operations)  
  - matplotlib & seaborn (For data visualization)  
  - category_encoders (For advanced encoding techniques)
- **Environment**: Jupyter Notebook

---

## 🚀 Getting Started

To run these notebooks on your local machine, follow these steps.

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Machine-learning-practice.git
cd machine-learning-practice
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv .venv
source .venv/Scripts/activate   # On Windows, use `.venv\Scripts\activate`
```

3. **Install the required dependencies:**
(Create a `requirements.txt` file with the content below)
```
pandas
numpy
matplotlib
seaborn
scikit-learn
category_encoders
```

Then run:
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

You can now navigate through the folders and run the notebooks!

---

## 📊 Example Output

Here is an example output of a **Decision Tree visualization**, generated from the `Decision_Tree.ipynb` notebook.

![Decision Tree](assets/licensed-image.jfif)

---

## 🔮 Future Plans

I plan to continuously expand this repository as my learning progresses:
- [ ] Add more advanced algorithms and ensemble methods (e.g., Gradient Boosting, AdaBoost).
- [ ] Incorporate Deep Learning concepts using PyTorch or TensorFlow.
- [ ] Work on end-to-end ML projects (e.g., Titanic Survival, House Price Prediction).
- [ ] Explore concepts in Generative AI (Gen AI).
