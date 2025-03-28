# Credit-Card-Fraud-Detection
A machine learning model for detecting fraudulent credit card transactions while minimizing false positives.

## Project Structure
```
credit-card-fraud-detection/
│-- data/                     # Directory for dataset files
│   ├── creditcard.csv        # Raw dataset
│   ├── preprocessed.csv      # Processed dataset
│-- notebooks/                # Jupyter notebooks for data analysis and experimentation
│   ├── EDA.ipynb             # Exploratory Data Analysis (EDA)
│   ├── model_training.ipynb  # Model training and evaluation
│-- src/                      # Source code directory
│   ├── data_preprocessing.py # Data cleaning and preprocessing script
│   ├── train_model.py        # Script for training machine learning model
│   ├── evaluate_model.py     # Model evaluation script
│-- results/                  # Directory to store results and reports
│   ├── model_performance.txt # Evaluation metrics
│   ├── confusion_matrix.png  # Confusion matrix visualization
│-- requirements.txt          # Dependencies and required packages
│-- README.md                 # Project documentation
│-- main.py                   # Main script to run fraud detection pipeline
```

## Introduction
Credit card fraud detection is a crucial application of machine learning to minimize financial losses. This project aims to build a classification model capable of distinguishing between fraudulent and non-fraudulent transactions with high accuracy.

## Dataset
- **Source:** Kaggle dataset "mlg-ulb/creditcardfraud"
- **Features:** Transaction amount, time, merchant details, and more.
- **Target:** Binary classification (0: Non-Fraud, 1: Fraud).

## Methodology
1. **Data Preprocessing:**
   - Handle missing values and outliers.
   - Standardize numerical features.
   - Address class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**.

2. **Exploratory Data Analysis (EDA):**
   - Visualize transaction distributions.
   - Analyze fraud vs. non-fraud transaction patterns.
   - Generate correlation heatmaps.

3. **Model Selection and Training:**
   - Used **Random Forest Classifier** for classification.
   - Applied feature scaling and hyperparameter tuning.
   - Split dataset into **80% training** and **20% testing**.

4. **Model Evaluation:**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)

## Usage
1. Place the dataset in the `data/` folder.
2. Run the preprocessing script:
   ```bash
   python src/data_preprocessing.py
   ```
3. Train the model:
   ```bash
   python src/train_model.py
   ```
4. Evaluate model performance:
   ```bash
   python src/evaluate_model.py
   ```

## Results
- Achieved **high accuracy** while minimizing false positives.
- Visualized the confusion matrix to analyze classification performance.
- Improved fraud detection with oversampling techniques.



