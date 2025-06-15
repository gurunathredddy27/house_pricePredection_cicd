import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


df = pd.read_csv(r'c:\Users\91944\AI-ML-DS_training\datasets\house_data.csv')

house = df.drop(['date', 'sqft_lot', 'view', 'condition', 'statezip', 'street', 'waterfront','sqft_above', 'sqft_basement','yr_renovated'], axis = 1)

house = pd.get_dummies(house, columns=['city','country'])

X = house.drop('price', axis = 1)
y = house['price']


X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.3, random_state=42)

#initialize the model
model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)
dt_model = DecisionTreeRegressor(random_state=42)
svm_model = SVR()

# train
model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

#predict
y_pred = model.predict(X_test)
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
svm_pred = svm_model.predict(X_test)


# print("Linear regression R² Score:", r2_score(y_test, y_pred))
# print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred))
# print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


def evaluate(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"--- {model_name} ---")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print()

evaluate(y_test, y_pred, "Linear Regression")
evaluate(y_test, rf_pred, "Random Forest Regressor")
evaluate(y_test, dt_pred, "Decision Tree Regressor")
evaluate(y_test, svm_pred, "Support Vector Regressor")


import pickle
pickle.dump(model, open('house.pkl', 'wb'))
pickle.dump(rf_model, open('house_rf.pkl', 'wb'))

features = X_train.columns.tolist()
pickle.dump(features, open('features.pkl', 'wb'))



print(type(model))
print(type(rf_model))

print("Models and features saved successfully.")


