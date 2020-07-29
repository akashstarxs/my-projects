import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import  files
uploaded = files.upload()
df = pd.read_csv("data.csv")

df.head()
df.shape

"""#count number of empty values

df.isna().sum()
"""

df = df.dropna(axis=1)

df.isna().sum()

df['diagnosis'].value_counts()

import seaborn as sns

sns.countplot(df['diagnosis'], label ='count')

#encoding categorical data values

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)

sns.pairplot(df.iloc[:,1:5],hue='diagnosis')

df.iloc[:,1:12].corr()

plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt='.0%')

#split dataset into independent(x)(desicison factors) and dependent (y)(desicion ) datset
x = df.iloc[:,2:31].values
y = df.iloc[:,1].values

#spliting traintest data
from sklearn.model_selection import train_test_split
X_trian,X_test,Y_trian,Y_test = train_test_split(x,y, test_size=0.25, random_state =0)

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X_trian = sc.fit_transform(X_trian)
X_test = sc.fit_transform(X_test)

#create a function for the models
def models(X_trian,Y_trian):

  #logistic regression
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression()
  log.fit(X_trian,Y_trian)

  #descision tree
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
  tree.fit(X_trian,Y_trian)

  #random Forest Clssifier
  from sklearn.ensemble import RandomForestClassifier
  forest =RandomForestClassifier(n_estimators =10, criterion ='entropy' , random_state = 0)
  forest.fit(X_trian,Y_trian)

  #accurary
  print('[0]LOgistic Regression :',log.score(X_trian,Y_trian))
  print('[0]Descicion tree :',tree.score(X_trian,Y_trian))
  print('[0]Random Forest :',forest.score(X_trian,Y_trian))

  return log,forest,tree

#get all models
model = models(X_trian,Y_trian)

#test model accuracy on test  data on confusion matrix

from sklearn.metrics import confusion_matrix
for i in range(len(model)):
  cm = confusion_matrix(Y_test, model[0].predict(X_test))
  TP = cm[0][0]
  TN = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]

  print(cm)
  print('Testing accuracy = ',(TP + TN)/(TP+TN+FN+FP))

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range(len(model)):
  print(classification_report(Y_test,model[0].predict(X_test)))
  print(accuracy_score(Y_test,model[0].predict(X_test)))
