import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


@st.cache_data
def load_data():
    df = pd.read_csv("Titanic-Dataset.csv")
    return df

df = load_data()

st.title("🚢 Titanic Survival Prediction App")

st.subheader("Dataset Preview")
st.dataframe(df.head())


# Drop less relevant columns
X = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Survived"], errors='ignore')
y = df["Survived"]

# Fill missing values and encode categorical data
for col in X.columns:
    if X[col].dtype == 'O':  # object = categorical
        X[col] = X[col].fillna("Missing")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    else:
        X[col] = X[col].fillna(X[col].median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)



st.subheader("🎯 Make a Prediction")

# Build dynamic input fields
user_input = {}
for col in X.columns:
    if df[col].dtype == 'O':
        options = list(df[col].dropna().unique())
        input_val = st.selectbox(f"{col}", options)
        le = LabelEncoder()
        le.fit(list(df[col].dropna().unique()))
        user_input[col] = le.transform([input_val])[0]
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_val = st.slider(f"{col}", min_val, max_val, mean_val)
        user_input[col] = input_val

user_df = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(user_df)[0]
    proba = model.predict_proba(user_df)[0]
    
    if prediction == 1:
        st.success(f"🎉 Prediction: Survived (Probability: {proba[1]*100:.2f}%)")
    else:
        st.error(f"⚠️ Prediction: Did NOT Survive (Probability: {proba[0]*100:.2f}%)")



st.subheader("📊 Data Exploration")

# Plot class balance
fig, ax = plt.subplots()
sns.countplot(x="Survived", data=df, ax=ax)
ax.set_title("Survival Count")
st.pyplot(fig)

# Plot feature importance
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

st.write("### Feature Importances")
st.dataframe(importance_df)

fig2, ax2 = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax2)
ax2.set_title("Feature Importance Plot")
st.pyplot(fig2)
