#program to detetct if a person has parkison's disease
#dataset available in kaggle :https://archive.ics.uci.edu/ml/datasets/Parkinsons

#get the dependencies
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

#load the data
from google.colab import files
uploaded = files.upload()

#loading data
df = pd.read_csv("parkinsons.data")

df.shape

#check data for missing values
df.isnull().values.any()

#get the target coloum
df['status'].value_counts()

#visualize count
sns.countplot(df['status'])

#create features
y = np.array(df['status'])
x = df.drop(['name'],1)
x = np.array(x.drop(['status'],1))

#split data into trian test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#create the XGBClassifier
model = XGBClassifier().fit(x_train,y_train)

#get the model prediction 
predictions = model.predict(x_test)
predictions

y_test

#model accuracy, precision, recall 
print(classification_report(y_test, predictions))

