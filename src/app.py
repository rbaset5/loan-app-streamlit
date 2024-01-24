from pickle import load
import streamlit as st
import os

if os.path.isfile("/workspaces/machine-learning-python-template-ds-2023/src/decision_tree_classifier_default_4.sav"):
    modelDir = "/workspaces/machine-learning-python-template-ds-2023/src/decision_tree_classifier_default_4.sav"
else:
    modelDir = "./decision_tree_classifier_default_4.sav"
model = load(open(modelDir, "rb"))
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

st.title("Iris - Model prediction")

val1 = st.slider("Petal width", min_value = 0.0, max_value = 4.0, step = 0.1)
val2 = st.slider("Petal length", min_value = 0.0, max_value = 4.0, step = 0.1)
val3 = st.slider("Sepal width", min_value = 0.0, max_value = 4.0, step = 0.1)
val4 = st.slider("Sepal length", min_value = 0.0, max_value = 4.0, step = 0.1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
