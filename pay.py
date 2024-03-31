# importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# loading the dataset

crop_data=pd.read_excel("C:\\Users\\DELL\\Desktop\\projectfinal\\pay_crops.xlsx")
# print(crop_data.columns)

crop_data=crop_data[crop_data['State_Name']=="Andhra Pradesh"]

print(crop_data.shape)

print(crop_data.columns)

print(crop_data.describe())

# Checking missing values of the dataset in each column

print(crop_data.isnull().sum())

# Dropping missing values

crop_data = crop_data.dropna()

#checking

print(crop_data.isnull().values.any())

crop_data.State_Name.unique()

# Adding a new column Yield which indicates Production per unit Area.

crop_data['Yield'] = (crop_data['Production'] / crop_data['Area'])

cd=crop_data
cd=cd.join(pd.get_dummies(cd['District_Name']))
cd=cd.join(pd.get_dummies(cd['Season']))
cd=cd.join(pd.get_dummies(cd['Crop']))
cd=cd.join(pd.get_dummies(cd['State_Name']))
print(cd.columns)

# Dropping unnecessary columns

cd= cd.drop(['State_Name'], axis = 1)
cd= cd.drop(['District_Name'], axis = 1)
cd= cd.drop(['Season'], axis = 1)
cd= cd.drop(['Crop'], axis = 1)
cd= cd.drop(['Production'], axis = 1)

#Training

# from sklearn.model_selection import train_test_split

x = cd.drop(["Yield"],axis=1)
y = cd["Yield"]

# Splitting data set - 25% test dataset and 75% train

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=5)
print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)
print(x.columns)

# from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# Initialize the decision tree regressor model
decision_tree_model = DecisionTreeRegressor()

# Initialize the random forest regressor model
random_forest_model = RandomForestRegressor(n_estimators=100)

# Initialize k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metric values for each model
dt_mae_scores = []
dt_mse_scores = []
dt_rmse_scores = []

rf_mae_scores = []
rf_mse_scores = []
rf_rmse_scores = []

# Perform k-fold cross-validation for Decision Tree and Random Forest
for train_idx, test_idx in kfold.split(x, y):
    # Split data into training and testing sets for this fold
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit the decision tree model on the training data for this fold
    decision_tree_model.fit(x_train, y_train)

    # Make predictions on the test data for this fold using Decision Tree
    dt_predictions = decision_tree_model.predict(x_test)

    # Compute evaluation metrics for Decision Tree
    dt_mae = mean_absolute_error(y_test, dt_predictions)
    dt_mse = mean_squared_error(y_test, dt_predictions)
    dt_rmse = np.sqrt(dt_mse)

    # Append metrics to lists for Decision Tree
    dt_mae_scores.append(dt_mae)
    dt_mse_scores.append(dt_mse)
    dt_rmse_scores.append(dt_rmse)

    # Fit the random forest model on the training data for this fold
    random_forest_model.fit(x_train, y_train)

    # Make predictions on the test data for this fold using Random Forest
    rf_predictions = random_forest_model.predict(x_test)

    # Compute evaluation metrics for Random Forest
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)

    # Append metrics to lists for Random Forest
    rf_mae_scores.append(rf_mae)
    rf_mse_scores.append(rf_mse)
    rf_rmse_scores.append(rf_rmse)

# Compute mean of metrics across all folds for Decision Tree
mean_dt_mae = np.mean(dt_mae_scores)
mean_dt_mse = np.mean(dt_mse_scores)
mean_dt_rmse = np.mean(dt_rmse_scores)

# Compute mean of metrics across all folds for Random Forest
mean_rf_mae = np.mean(rf_mae_scores)
mean_rf_mse = np.mean(rf_mse_scores)
mean_rf_rmse = np.mean(rf_rmse_scores)

# Print evaluation metrics for Decision Tree
# print("Evaluation metrics for Decision Tree:")
# print("Mean Absolute Error (MAE):", mean_dt_mae)
# print("Mean Squared Error (MSE):", mean_dt_mse)
# print("Root Mean Squared Error (RMSE):", mean_dt_rmse)

# # Print evaluation metrics for Random Forest
# print("\nEvaluation metrics for Random Forest:")
# print("Mean Absolute Error (MAE):", mean_rf_mae)
# print("Mean Squared Error (MSE):", mean_rf_mse)
# print("Root Mean Squared Error (RMSE):", mean_rf_rmse)

if mean_dt_mse > mean_rf_mse:
    pickle.dump(random_forest_model,open('pay.pkl','wb'))
else:
    pickle.dump(decision_tree_model,open('pay.pkl','wb'))

print("end of project")