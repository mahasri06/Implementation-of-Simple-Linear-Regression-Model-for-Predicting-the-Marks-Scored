# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### step 1 :Import the standard Libraries. 
#### step 2 :Set variables for assigning dataset values. 
#### step 3 :Import linear regression from sklearn. 
#### step 4 :Assign the points for representing in the graph. 
#### step 5 :Predict the regression for marks by using the representation of the graph. 
#### step 6 :Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
### MAHASRI P (212223100029)
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
## Dataset:
![dataset](https://github.com/charumathiramesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120204455/c7816d33-6dab-45e2-8d19-9a11e9583cb5)

## Head values:
![head](https://github.com/charumathiramesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120204455/7f3d7783-4601-4e70-989f-2ccbf87d0765)

## Tail values:
![tail](https://github.com/charumathiramesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120204455/5343e114-fe3a-4ad7-8058-6b81db462fdc)
 

## X and Y values:
![xyvalues](https://github.com/charumathiramesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120204455/f84947e0-99a3-444c-8286-c59cc0660a4e)


 ## Predication values of X and Y:
![predict ](https://github.com/charumathiramesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120204455/6ea46100-8530-4491-821e-079308a1eef5)

 ## MSE,MAE and RMSE:
![values](https://github.com/charumathiramesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120204455/0f3750f1-fec0-4008-abcf-7e7b971d82a9)


 ## Training Set:
![263971254-4b3957fd-3a36-4d22-ba6e-cb7204cd2f84](https://github.com/charumathiramesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120204455/088c3714-a70d-4ef0-b952-1d26c48e1fa8)


 ## Testing Set:

![263970887-0f5b3275-abac-4d48-a56f-b7634d559a81](https://github.com/charumathiramesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120204455/aa18e6a5-11f7-410e-bbd6-89c052ff52a6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
