# 🩺 SVM Model for Diabetes Prediction

## 📋 Project Overview

This project uses **Support Vector Machine (SVM)** to predict diabetes using the Pima Indians Diabetes Database from Kaggle.

---

## 🚀 Step-by-Step Implementation

### (a) 📚 Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
```

**What it does:** Load all necessary tools for data handling, model building, and visualization.

---

### (b) 📂 Import Dataset

```python
df = pd.read_csv("diabetes.csv")
df.head()
```

**What it does:** Read the diabetes dataset from CSV file and display the first few rows.

---

### (c) 🔍 Check Missing Values

```python
print(df.isnull().sum())
```

**What it does:** Count missing (null) values in each column to identify data quality issues.

---

### (d) 🛠️ Feature Engineering

```python
zero_value_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_value_columns] = df[zero_value_columns].replace(0, np.nan)
df.fillna(df.median(), inplace=True)
```

**What it does:** Replace impossible zero values with median values to improve data quality.

**Why?** Medical measurements like glucose and blood pressure can't be zero - these are missing data!

---

### (e) ✂️ Split the Dataset

```python
x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
```

**What it does:**

- Separate features (X) from target (y)
- Split data: 80% training, 20% testing
- `stratify=y` ensures balanced distribution

---

### (f) 🤖 Build SVM Model

```python
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train, y_train)
```

**What it does:**

1. **Standardize** features (mean=0, std=1) - SVM needs this!
2. Create SVM with **linear kernel**
3. **Train** the model on training data

---

### (g) 🎯 Assess the Accuracy

```python
y_pred = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nSVM Model Accuracy: {accuracy * 100:.2f}%")
```

**What it does:** Make predictions on test data and calculate how often the model is correct.

---

### (h) 📊 Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix Heatmap')
plt.show()
```

---

## 🎓 Learning Outcomes

✔️ Data preprocessing techniques  
✔️ Handling missing/invalid data  
✔️ SVM classification  
✔️ Model evaluation metrics  
✔️ Confusion matrix interpretation

---

**Happy Learning! 🎉**
