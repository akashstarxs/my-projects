import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("heart.csv")

y = np.array(df['DEATH_EVENT'])
x = df.drop(['DEATH_EVENT'],1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =.20)

sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from xgboost import XGBClassifier

model = XGBClassifier().fit(x_train,y_train)

predictions = model.predict(x_test)
predictions

y_test

print(classification_report(y_test, predictions))
