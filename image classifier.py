#this project was executed using colab
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#load data
from keras.datasets import cifar10
(X_trian,Y_trian),(X_test,Y_test) = cifar10.load_data()

#image as array
index = 0
X_trian[index]

#image as picture
img = plt.imshow(X_trian[index])
print(Y_trian[index])

#classifier
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

#convert the labels into set of 10 numbers to input into the network

Y_trian_one_hot = to_categorical(Y_trian)
Y_test_one_hot = to_categorical(Y_test)

X_trian = X_trian/255
Y_trian = Y_trian/255

#creating the model

model = Sequential()

#layers
model.add(Conv2D(32,(5,5),activation='relu',input_shape=(32,32,3)))
#pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))
#layer3
model.add(Conv2D(32,(5,5),activation='relu'))
#layer 4
model.add(MaxPooling2D(pool_size = (2,2)))
#layer 5
model.add(Flatten())
#layer 6 with 1000 neurons
model.add(Dense(1000,activation='relu'))
#layer7 Dropout
model.add(Dropout(0.5))
#layer 7 with 500 neurons
model.add(Dense(500,activation='relu'))
#layer9 Dropout
model.add(Dropout(0.5))
#layer10 with 250 neurons
model.add(Dense(250,activation='relu'))
#layer 11 with 10
model.add(Dense(10,activation='softmax'))

#compiling the model
model.compile(loss = 'categorical_crossentropy',
              optimizer ='adam',
              metrics = ['accuracy'])

#trian the model
hist = model.fit(X_trian,Y_trian_one_hot,
                 batch_size = 256,
                 epochs =10,
                 validation_split =0.2)

#evaluate the model using test data
model.evaluate(X_test,Y_test_one_hot)[1]

#accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc = 'upper right')
plt.show()

#loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'],loc = 'upper right')
plt.show()

#test the model....comment this portion if trianing
from google.colab import files
uploaded = files.upload()

new_image = plt.imread('horse.jpg')
img = plt.imshow(new_image)

#resize image
from skimage.transform import resize
resized_image = resize(new_image,(32,32,3))
img = plt.imshow(resized_image)

predictions = model.predict(np.array([resized_image]))
#show the prediction
predictions

#sort the prediction from least to greatest
list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
  for j in range(10):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp

#show perdictions
print(list_index)

#print most likely class
for i in range(5):
  print(classification[list_index[i]],':',predictions[0][list_index[i]]*100,%)
