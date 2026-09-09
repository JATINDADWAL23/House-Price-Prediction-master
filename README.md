# House Price Prediction AI/ML Project

This project involves building a machine learning model to predict house prices based on various features. The dataset used for this project is from the Kaggle competition "House Prices - Advanced Regression Techniques". The goal is to develop a model that accurately predicts house prices given a set of input features.

## Kaggle Competition
- Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Model Score: 87.16% (R-squared score)

## File Structure
- `house_price_prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and prediction.
- `submission.csv`: CSV file containing the predicted house prices for the test dataset.
- `gbr.pkl`: Pickle file containing the trained GradientBoostingRegressor model.

## Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Data Loading and Analysis
- The training and test datasets are loaded from CSV files.
- Exploratory data analysis is performed to understand the structure and characteristics of the data.
- Data visualization techniques such as histograms, box plots, and heatmaps are used to analyze the distribution of features and identify missing values.

## Data Preprocessing
- Missing values are handled using appropriate techniques such as imputation or dropping columns.
- Categorical variables are encoded using one-hot encoding.
- Numerical features are standardized to ensure uniformity and improve model performance.

## Model Selection and Training
- Several regression models are considered, including Linear Regression, SVR, SGDRegressor, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, and MLPRegressor.
- Cross-validation is used to evaluate each model's performance based on the R-squared score.
- The GradientBoostingRegressor model is selected based on its superior performance.

## Model Evaluation and Prediction
- The selected model is trained on the training dataset.
- The trained model is used to make predictions on the test dataset.
- The predictions are saved to a CSV file (`submission.csv`) for submission.


## Additional Notes
- The `submission.csv` file contains the predicted house prices for the test dataset.
- The trained model (`gbr.pkl`) is stored as a pickle file for future use or deployment.

For any further inquiries or improvements, feel free to reach out.

## Run the application

The project now includes a Streamlit frontend and a Python prediction backend.

```powershell
cd "C:\Users\jatin\Downloads\yyf\House-Price-Prediction-master"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser. The model trains from
`ML_Model/data_set/train.csv` and uses the form to estimate a house sale price.
The estimate separates house and land: the house depreciates by 5% per year,
while land appreciates according to the selected urban, rural, or high-growth
land category. The displayed total is converted at ₹91 per US dollar.
The form also asks for total rooms, average room size, and number of floors. It
calculates a separate room-wise construction estimate: ₹1,500/sq ft for a
small two-room house, ₹2,400/sq ft for a large multi-floor house with large
rooms, and ₹1,950/sq ft for other cases.
