
from google.colab import files
import pandas as pd

df = pd.read_csv("RTA Dataset.csv")
df.head()
df.info()
df.describe(include='all')
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
import seaborn as sns
import matplotlib.pyplot as plt
target = 'Accident_severity'
features = df.drop(columns=[target])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df[target] = le.fit_transform(df[target])

df_encoded = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, df[target], test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
import numpy as np
new_data = np.array([scaled_features[0]])  # sample input
model.predict(new_data)
def preprocess_input(input_dict):
    df_input = pd.DataFrame([input_dict])
    df_input_encoded = pd.get_dummies(df_input)
    df_input_encoded = df_input_encoded.reindex(columns=df_encoded.columns, fill_value=0)
    scaled_input = scaler.transform(df_input_encoded)
    return scaled_inputdef predict_severity(input_dict):
    input_processed = preprocess_input(input_dict)
    pred = model.predict(input_processed)
    return le.inverse_transform(pred)[0]
def predict_interface(Sex_of_driver, Age_band_of_driver, Type_of_vehicle, Cause_of_accident):
    input_dict = {
        "Sex_of_driver": Sex_of_driver,
        "Age_band_of_driver": Age_band_of_driver,
        "Type_of_vehicle": Type_of_vehicle,
        "Cause_of_accident": Cause_of_accident,
        # Add other required fields with default values or inputs
    }
    return predict_severity(input_dict)
pip install gradio
import gradio as gr

interface = gr.Interface(
    fn=predict_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Driver Gender"),
        gr.Dropdown(["18-30", "31-50", "Above 51"], label="Driver Age"),
        gr.Dropdown(df["Type_of_vehicle"].dropna().unique().tolist(), label="Vehicle Type"),
        gr.Dropdown(df["Cause_of_accident"].dropna().unique().tolist(), label="Accident Cause"),
    ],
    outputs="text"
)
import gradio as gr

interface = gr.Interface(
    fn=predict_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Driver Gender"),
        gr.Dropdown(["18-30", "31-50", "Above 51"], label="Driver Age"),
        gr.Dropdown(df["Type_of_vehicle"].dropna().unique().tolist(), label="Vehicle Type"),
        gr.Dropdown(df["Cause_of_accident"].dropna().unique().tolist(), label="Accident Cause"),
    ],
    outputs="text"
)

interface.launch()

