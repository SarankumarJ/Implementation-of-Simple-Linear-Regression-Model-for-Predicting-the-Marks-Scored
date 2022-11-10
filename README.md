# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for Gradient Design.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```py
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sarankumar J
RegisterNumber: 212221230087

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
#segregation data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#spliting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted value
Y_pred
#displaying actual value
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(Y_test,Y_pred)
print('MSC=',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)

rmse=np.sqrt(mse)
print("RMSE=",rmse)
```

## Output:
![image](https://user-images.githubusercontent.com/94778101/200994040-c8267b66-c706-49f0-a890-8e4dd96838b6.png)
![image](https://user-images.githubusercontent.com/94778101/200994097-6ba544fd-f86a-4632-9a01-02cf9e5ac129.png)
![image](https://user-images.githubusercontent.com/94778101/200994118-2577ed97-cd2a-4589-a90f-7653d3b9691a.png)
![image](https://user-images.githubusercontent.com/94778101/200994190-28de1367-e0a0-463b-a93a-bd948cb35850.png)
![image](https://user-images.githubusercontent.com/94778101/200994202-8ba7ff34-e479-4593-af8e-50b13dbdd69c.png)
![image](https://user-images.githubusercontent.com/94778101/200994230-f561bdc8-91d1-48c1-a5f1-1ab42c6e5133.png)
![image](https://user-images.githubusercontent.com/94778101/200994256-4493e8b5-ae82-47bb-a0b5-83ef428d28a6.png)
![image](https://user-images.githubusercontent.com/94778101/200994280-8b358bd7-5ebb-4521-9c69-1937553bb4a0.png)
![image](https://user-images.githubusercontent.com/94778101/200994319-6899499b-04b9-4ad5-b107-82b77a3b7c0e.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
