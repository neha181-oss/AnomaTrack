# AnomaTrack: Anomaly Detection in Stock Data

**AnomaTrack** is an AI-powered anomaly detection system for stock data. It identifies unusual sales patterns using Z-score analysis and provides AI-generated explanations for the anomalies. With interactive visualizations and detailed insights, it helps businesses detect and understand outliers in sales data effectively.

## Features:
- **Anomaly Detection**: Uses Z-score to identify anomalies in sales data.
- **AI-Generated Explanations**: Offers plausible explanations for detected anomalies using large language models (LLMs).
- **Interactive Visualization**: Visualizes anomalies using interactive scatter plots.
- **Summary Report**: Displays a summary of detected anomalies and their explanations.

## Usage:
1. Run the app using Streamlit.
2. The app will generate simulated sales data, detect anomalies, and provide AI-generated comments on the detected anomalies.
3. Anomalies are visualized with a color-coded scatter plot.

## How It Works:
1. **Generate Sales Data**: The app simulates a dataset with sales data, including some anomalies.
2. **Detect Anomalies**: The Z-score is calculated for each sales record to identify anomalies.
3. **AI Explanation**: For each anomaly, an AI model generates an explanation for why it might be an anomaly.
4. **Visualization**: The data is displayed using interactive visualizations, where anomalies are highlighted.

## Example:
In the dataset, anomalies may include extremely high sales amounts that are unlikely to occur naturally, and the AI will generate explanations such as "an unusually high order due to a bulk purchase" or "an error in data entry."

## Acknowledgments:
- **Streamlit**: For building the app interface.
- **Plotly**: For interactive visualizations.
- **Hugging Face**: For providing the language model for anomaly explanation.
- **NumPy** and **Pandas**: For data handling and manipulation.
