import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
data=pd.read_csv("/content/titanic.csv.csv")
data.head(10)
data=data.drop(['PassengerId','Sex','Name','Cabin','Embarked','Cabin','Ticket'],axis=1)
data.isnull().sum()
data['Age'].fillna(data['Age'].mean(),inplace=True)
data['Fare'].fillna(data['Fare'].mean(),inplace=True)
X=data.drop(['Survived'],axis=1)
y=data['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=25,test_size=0.1)
print(X.shape,X_train.shape,X_test.shape)
 model=svm.SVC(kernel='linear')
 model.fit(X_train,y_train)
 y_pred=model.predict(X_test)
 Raccuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x=data['Survived'],data=data)
plt.figure(figsize=(10,6))
sns.boxplot(x='Survived',y='Age',data=data,palette='winter')
plt.show
