# Housing Price Prediction

## Project Overview
This project aims to predict house prices using machine learning models. The dataset consists of numerical and categorical features that influence housing prices. The workflow includes data preprocessing, feature engineering, model training, and evaluation to determine the best-performing model.

## Steps Followed

### 1️⃣ Data Preparation
- Imported essential libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`).
- Loaded the dataset and examined its structure.
- Identified feature types (numerical, categorical) and handled missing values.

### 2️⃣ Exploratory Data Analysis (EDA)
- Plotted scatterplots to identify trends and outliers.
- Generated a correlation heatmap to find the most important features.
- Analyzed the `SalePrice` distribution to check for skewness.

### 3️⃣ Feature Engineering
Created new features to enhance model performance:

- `HouseAge = YrSold - YearBuilt` (Age of house since built)
- `HouseRemodelAge = YrSold - YearRemodAdd` (Years since last remodel)
- `TotalSF = 1stFlrSF + 2ndFlrSF + BsmtFinSF1 + BsmtFinSF2` (Total square footage)
- `TotalArea = GrLivArea + TotalBsmtSF` (Total living + basement area)
- `TotalBaths = BsmtFullBath + FullBath + 0.5 * (BsmtHalfBath + HalfBath)` (Total bathrooms, where half baths count as 0.5)
- `TotalPorchSF = OpenPorchSF + 3SsnPorch + EnclosedPorch + ScreenPorch + WoodDeckSF` (Total porch and deck space)
- Dropped irrelevant or highly correlated features.

### 4️⃣ Data Preprocessing
#### Categorical Features
- Ordinal features → Encoded using `OrdinalEncoder`.
- Nominal features → Applied `OneHotEncoding`.

#### Numerical Features
- Handled missing values.
- Standardized using `StandardScaler`.
- Combined all transformations using `ColumnTransformer`.

### 5️⃣ Target Variable Transformation
- Applied log transformation: `np.log1p(SalePrice)` to normalize the target variable.

### 6️⃣ Model Training & Evaluation
Split data into **80% training / 20% testing** and trained multiple models:
✅ Linear Regression  
✅ Random Forest Regressor (with `GridSearchCV` for hyperparameter tuning)  
✅ XGBoost  
✅ Ridge Regression  
✅ Gradient Boosting Regressor  
✅ LGBM Regressor (with Cross Validation)  
✅ CatBoost Regressor  

**Evaluation Metric:** Root Mean Squared Error (RMSE)

### 7️⃣ Stacking & Voting Regressor for Final Prediction
- Implemented **Voting Regressor** to combine multiple models based on their performance.
- Implemented **Stacking Regressor**, leveraging multiple algorithms to enhance predictive power.
- Selected **Voting Regressor** as the best-performing model.
- Generated final predictions on the test set.

### 8️⃣ Submission
- Converted predictions back to the original scale using `np.exp()`.
- Created a submission file (CSV) containing the predicted `SalePrice`.

## 🔥 Feature Importance Analysis (Post-Prediction Recommendation)
Feature importance analysis helps determine which **input variables (features) have the most impact** on the target variable in a predictive model. This analysis enhances model interpretability and can guide future feature selection. The histogram below shows the **top 30 important features** based on their impact.

## 📊 Best Model Selection
### From the Comparison Table, **Voting Regressor** outperformed all models in accuracy and stability, as it has:
- **Lowest RMSE** (TBD) → Indicates minimal prediction error.
- **Highest R² Score** (TBD) → Shows strong predictive power.
- **Lowest MAPE** (TBD%) → Suggests better relative accuracy compared to others.

## 📂 Technologies Used
- **Python** (`pandas`, `numpy`, `matplotlib`, `seaborn`)
- **Scikit-learn** (`LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`, `GridSearchCV`)
- **XGBoost**, **CatBoost**, **LGBM**
- **Feature Engineering & Data Preprocessing** (`ColumnTransformer`, `OneHotEncoding`, `StandardScaler`)

## 📌 Conclusion
This project demonstrated the effectiveness of machine learning in predicting house prices. By leveraging multiple regression models and feature engineering, a highly accurate predictive model was developed. The insights gained can help real estate professionals and buyers make **data-driven** decisions based on model predictions.

 

## **📂 Technologies Used**  
- **Python** (`pandas`, `numpy`, `matplotlib`, `seaborn`)  
- **Scikit-learn** (`LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`, `GridSearchCV`)  
- **XGBoost, CatBoost, LGBM**  
- **Feature Engineering & Data Preprocessing** (`ColumnTransformer`, `OneHotEncoding`, `StandardScaler`)  
