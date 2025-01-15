import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import plotly.express as px
from transformers import pipeline

# Function to generate sales data with anomalies
def generate_sales_data_with_anomalies():
    dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq='D')

    # Generate normal data
    sheet1_data = {
        "Date": np.random.choice(dates, 200),
        "Product_ID": np.random.randint(1000, 1020, 200),
        "Product_Name": np.random.choice(['Product_A', 'Product_B', 'Product_C'], 200),
        "Category": np.random.choice(['Electronics', 'Furniture', 'Clothing'], 200),
        "Quantity_Sold": np.random.randint(1, 10, 200),
        "Sales_Amount": np.random.randint(100, 1000, 200)
    }
    df1 = pd.DataFrame(sheet1_data)

    # Introduce anomalies with very high Sales_Amount values
    anomaly_indices = np.random.choice(df1.index, size=5, replace=False)
    df1.loc[anomaly_indices, 'Sales_Amount'] = df1['Sales_Amount'].max() * 10  # Set sales amount 10 times higher

    return df1

# Detect anomalies using Z-score
def detect_anomalies(df):
    df['Z_Score'] = (df['Sales_Amount'] - df['Sales_Amount'].mean()) / df['Sales_Amount'].std()
    df['Anomaly'] = df['Z_Score'].apply(lambda x: 'Anomaly' if abs(x) > 2 else 'Normal')
    return df

# Generate comments for anomalies using an LLM
def generate_comments_with_llm(anomalies_df):
    # Load Hugging Face LLM pipeline
    llm = pipeline("text-generation", model="distilgpt2")

    comments = []
    for _, row in anomalies_df.iterrows():
        prompt = (
            f"Explain why the following sales record is an anomaly:\n"
            f"Date: {row['Date'].date()}, Product: {row['Product_Name']}, "
            f"Category: {row['Category']}, Quantity Sold: {row['Quantity_Sold']}, "
            f"Sales Amount: ${row['Sales_Amount']}, Z-Score: {row['Z_Score']:.2f}.\n"
            f"Provide a plausible reason."
        )
        response = llm(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        comments.append(response)

    anomalies_df['Comments'] = comments
    return anomalies_df

# Streamlit App
def main():
    st.title("Sales Data Anomaly Detection with AI-Generated Comments")
    
    # Generate and display sales data
    df_sales = generate_sales_data_with_anomalies()
    st.write("### Generated Sales Data")
    st.dataframe(df_sales)

    # Detect anomalies
    df_sales = detect_anomalies(df_sales)
    st.write("### Sales Data with Anomalies Highlighted")
    st.dataframe(df_sales)

    # Anomaly summary
    num_anomalies = df_sales['Anomaly'].value_counts().get('Anomaly', 0)
    total_records = len(df_sales)
    st.write(f"### Summary")
    st.write(f"In this dataset of {total_records} records, {num_anomalies} anomalies were detected.")
    
    # Visualization
    st.write("### Anomaly Visualization")
    fig = px.scatter(df_sales, x="Date", y="Sales_Amount", color="Anomaly",
                     title="Sales Amount with Anomalies",
                     labels={"Sales_Amount": "Sales Amount ($)", "Date": "Date"},
                     color_discrete_map={"Anomaly": "red", "Normal": "blue"})
    st.plotly_chart(fig)

    # Filter anomalies and generate AI comments
    st.write("### Filtered Anomalies with AI-Generated Comments")
    anomalies_df = df_sales[df_sales['Anomaly'] == 'Anomaly']
    anomalies_df = generate_comments_with_llm(anomalies_df)
    st.dataframe(anomalies_df[['Date', 'Product_ID', 'Product_Name', 'Sales_Amount', 'Z_Score', 'Comments']])

if __name__ == "__main__":
    main()
