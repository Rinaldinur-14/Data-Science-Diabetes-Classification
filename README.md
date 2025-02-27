# Diabetes Progression Prediction with Machine Learning Models

## 📚 **Project Description**
This project aims to predict the progression of diabetes in patients using a **Random Forest Regressor**. The dataset used is the [Diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) from Scikit-learn. This dataset contains 10 baseline variables (features) collected from 442 diabetes patients, along with a quantitative measure of disease progression one year after baseline.

---

## 🎯 **Goals**
1. **Understand the dataset**: Perform exploratory data analysis (EDA) to understand the distribution of features and their relationships with the target variable.
2. **Build and train a regression model**: Use a Random Forest Regressor to predict diabetes progression.
3. **Evaluate the model's performance**: Use metrics like Mean Squared Error (MSE) and R-squared (R²) to assess the model's accuracy.
4. **Visualize the results**: Plot the actual vs. predicted values to understand the model's performance.

---

## 📊 **Dataset Overview**
The Diabetes dataset consists of **442 samples** and **10 features**. Each feature represents a medical measurement, and the target variable is a quantitative measure of disease progression.

### **Features**:
1. **Age**: Age of the patient (scaled).
2. **Sex**: Gender of the patient (scaled).
3. **BMI**: Body Mass Index (scaled).
4. **BP**: Average blood pressure (scaled).
5. **S1**: Total serum cholesterol (scaled).
6. **S2**: Low-density lipoproteins (scaled).
7. **S3**: High-density lipoproteins (scaled).
8. **S4**: Total cholesterol / HDL ratio (scaled).
9. **S5**: Log of serum triglycerides level (scaled).
10. **S6**: Blood sugar level (scaled).

### **Target**:
- **Disease Progression**: A quantitative measure of diabetes progression one year after baseline.

---

## ⚙️ **Algorithm Used**
The primary algorithm used in this project is:
- **[Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)**: An ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## 🧠 **Insight**
From the analysis, the following insights were obtained:
- **Feature Importance**: Features like `bmi` (Body Mass Index) and `s5` (a blood serum measurement) were found to be the most significant predictors of diabetes progression.
- **Model Performance**: The Random Forest Regressor achieved an **R² score of 0.44**, indicating that the model explains 44% of the variance in the target variable. The **Mean Squared Error (MSE)** was **2859.69**, suggesting moderate prediction accuracy.
- **Visualization**: The scatter plot of actual vs. predicted values shows that the model performs reasonably well, but there is room for improvement, especially for higher target values.

---

## 🛠️ **Dependencies**
To run this project locally, you need to install the following Python dependencies:
- 📚 [Scikit-learn](https://scikit-learn.org/stable/index.html): For machine learning tools and datasets.
- 📊 [matplotlib](https://matplotlib.org/): For data visualization.
- 🔢 [numpy](https://numpy.org/): For numerical computations.
- 📋 [pandas](https://pandas.pydata.org/): For data manipulation and analysis.
- 🌲 [seaborn](https://seaborn.pydata.org/): For advanced statistical visualizations.


