# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries (pandas, chardet, sklearn, etc.).
2.Detect the encoding of the CSV file using chardet.
3.Read the CSV file with the correct encoding.
4.Check the data for structure and missing values.
5.Split the data into input (x = messages) and output (y = labels).
6.Divide the data into training and testing sets.
7.Divide the data into training and testing sets.
8.Train an SVM model, make predictions, and calculate accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Thamizh S
RegisterNumber:  212224040350
*/
import chardet
file='spam.csv'
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy 
*/
```

## Output:
## Result output

<img width="1022" height="35" alt="image" src="https://github.com/user-attachments/assets/f9089f7a-b595-402d-b7d5-f61a22a4ce9b" />

## data.head()

<img width="1024" height="295" alt="image" src="https://github.com/user-attachments/assets/596eae19-cf9b-42a1-a6d2-aca38e8a7a69" />

## data.info()

<img width="599" height="401" alt="image" src="https://github.com/user-attachments/assets/c77ccb3f-5888-4047-a403-3fbb01f62964" />

## data.isnull().sum()

<img width="293" height="200" alt="image" src="https://github.com/user-attachments/assets/d6e816aa-2a42-46aa-8588-02ce2a55de1b" />

## y_pred

<img width="984" height="44" alt="image" src="https://github.com/user-attachments/assets/0b7f0394-bf58-4b09-8cb7-305575b02134" />


## accuracy()
<img width="329" height="56" alt="image" src="https://github.com/user-attachments/assets/f3c3ee10-c6c4-4f7c-a76a-4824da49005e" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
