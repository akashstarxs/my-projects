#libraries
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st 

#create a title and subtitle
st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and python """)

#open and display image
image = Image.open('genetics.jpg')
st.image(image, caption='ML', use_column_width=True)

#trianing data
df = pd.read_csv('diabetes.csv')

st.subheader('Data Information')

#data
st.dataframe(df)
#statistics
st.write(df.describe())
#data visuals
st.bar_chart(df)

#spliting data into dependent and independent variables
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values
#split the data set into trianing and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#get feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies',0,17,3)
    glucose = st.sidebar.slider('Glucose',0,199,117)
    blood_pressure = st.sidebar.slider('Blood _prussure',0,122,73)
    skin_thickness = st.sidebar.slider('skin_thickness',0,99,23)
    insulin = st.sidebar.slider('Insulin',0.0,846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    diabetes_pedigree_function = st.sidebar.slider('dpf',0.078,2.42,0.3725)
    Age = st.sidebar.slider('Age',21,81,29)
    
    #store a dictonary  into variable
    user_data = {'pregnancies': pregnancies,
                 'glucose':glucose,
                 'blood_pressure':blood_pressure,
                 'skin_thickness':skin_thickness,
                 'insulin':insulin,
                 'BMI':BMI,
                 'diabetes_pedigree_function':diabetes_pedigree_function,
                 'age':Age
                 }
    #transform data into dataframe
    features = pd.DataFrame(user_data,index=[0])
    return features


#store the users input into a variable
user_input = get_user_input()

#set subheader
st.subheader('User Input')
st.write(user_input)

#model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)

#show the model metrics
st.subheader('Model Test accuracy score')
st.write(_str(accuracy_score(y_test,  RandomForestClassifier.predict(x_test)) * 100)+'%')

#store the model predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

#subheader
st.subheader('Classification')
st.write(prediction)



