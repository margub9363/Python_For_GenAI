import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names


df,target_name = load_data()
model=RandomForestClassifier()
model.fit(df.iloc[:,:-1], df['species'])

st.sidebar.title('Input Features')
sepal_length = st.sidebar.slider("Sepal length", float(df['sepal length (cm)'].min()),  float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal width", float(df['sepal length (cm)'].min()),  float(df['sepal length (cm)'].max()))
petal_length = st.sidebar.slider("Petal length", float(df['sepal length (cm)'].min()),  float(df['sepal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal width", float(df['sepal length (cm)'].min()),  float(df['sepal length (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]


# Prediction
prediction = model.predict(input_data)
predicted_species = target_name[prediction[0]]


# ---- MAIN PAGE OUTPUT ----
st.title("Iris Flower Classifier")

st.subheader("Your Input")
st.write({
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width,
})

st.subheader("Prediction")
st.success(f"Predicted species: **{predicted_species}**")