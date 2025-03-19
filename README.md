# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format(e.g.,CSV,DataFrame)
2. Use a Simple Linear Regression model to fit the training data.
3. Use the trained model to predict values for the test set.
4. Evaluate performanceusing metrics like Mean Absolute Error(MAE) and Root Mean Squared Error(RMSE)

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Meenakshi.R
RegisterNumber:212224220062
```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datfile
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:

![image](https://github.com/user-attachments/assets/19bc5cd3-1a23-4690-932e-bb2ca716597a)

![image](https://github.com/user-attachments/assets/f2eaed5d-4b02-42c0-86b1-154945fa3ec7)

![image](https://github.com/user-attachments/assets/bb797617-4600-4f33-9ab9-6e7c5c5be676)

![image](https://github.com/user-attachments/assets/bb309010-eb20-4d2d-8215-f98a70a6fa93)

![image](https://github.com/user-attachments/assets/663a9968-bd65-48b7-b296-a1f5c8af474d)

![image](https://github.com/user-attachments/assets/0792c9e7-9b02-48f8-b83e-10fbc273db45)

![image](https://github.com/user-attachments/assets/7ecefef8-a779-4045-88ed-441761c18b2c)

![image](https://github.com/user-attachments/assets/e048ba53-175b-4078-a7a5-3b222590064f)

![image](https://github.com/user-attachments/assets/3fc75420-24be-405c-aaf0-6b7445c094c9)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
