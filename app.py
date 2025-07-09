import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import openpyxl

st.set_page_config(page_title="Smart Business Dashboard", layout="wide")
st.title("🧠 AI-Powered Business Dashboard with Forecasting")

uploaded_file = st.sidebar.file_uploader("Upload Online Retail.xlsx", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Online Retail")
    df.dropna(subset=['CustomerID'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    df['Date'] = df['InvoiceDate'].dt.date

    st.sidebar.header("🔍 Filters")
    countries = st.sidebar.multiselect("Select Countries", df['Country'].unique(), default=["United Kingdom"])
    df = df[df['Country'].isin(countries)]

    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    st.sidebar.markdown("✅ Use filters above to interact with data")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"£{df['Sales'].sum():,.2f}")
    col2.metric("Total Orders", df['InvoiceNo'].nunique())
    col3.metric("Unique Customers", df['CustomerID'].nunique())

    sales_daily = df.groupby('Date')['Sales'].sum().reset_index()
    fig = px.line(sales_daily, x='Date', y='Sales', title="📈 Daily Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

    top_products = df.groupby('Description')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(top_products, x='Sales', y='Description', orientation='h', title="🏆 Top 10 Products")
    st.plotly_chart(fig2, use_container_width=True)

    country_sales = df.groupby('Country')['Sales'].sum().sort_values(ascending=False).reset_index()
    fig3 = px.bar(country_sales, x='Country', y='Sales', title="🌍 Sales by Country", color='Sales')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🔮 Predict Future Sales")
    forecast_days = st.slider("Forecast Days", 7, 60, 14)

    sales_daily['Days_Since'] = (pd.to_datetime(sales_daily['Date']) - pd.to_datetime(sales_daily['Date'].min())).dt.days
    model = LinearRegression()
    X = sales_daily[['Days_Since']]
    y = sales_daily['Sales']
    model.fit(X, y)

    future_days = np.arange(X['Days_Since'].max() + 1, X['Days_Since'].max() + forecast_days + 1).reshape(-1, 1)
    predictions = model.predict(future_days)

    future_dates = pd.date_range(sales_daily['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': predictions})

    fig4 = px.line(prediction_df, x='Date', y='Predicted Sales', title=f"📊 {forecast_days}-Day Sales Forecast")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📄 Raw & Forecast Data")
    st.dataframe(df)

    csv = prediction_df.to_csv(index=False).encode()
    st.download_button("📥 Download Forecast CSV", csv, "sales_forecast.csv", "text/csv")

else:
    st.info("Upload your `Online Retail.xlsx` file to get started.")
