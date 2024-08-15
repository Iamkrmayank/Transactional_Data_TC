import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to display demographic visualizations
def demographic_visualizations(df):
    st.title('Demographic Data Visualization')

    # Age-Based Segmentation Visualization
    if 'Age' in df.columns:
        st.subheader('Age-Based Segmentation Visualization')
        age_segment_counts = {
            'Younger (18-25)': len(df[(df['Age'] >= 18) & (df['Age'] <= 25)]),
            'Middle-Aged (26-45)': len(df[(df['Age'] >= 26) & (df['Age'] <= 45)]),
            'Older (46+)': len(df[df['Age'] >= 46])
        }
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(age_segment_counts.keys()), y=list(age_segment_counts.values()), palette='Blues_d')
        plt.title('Age-Based Segmentation of Customers')
        plt.xlabel('Age Group')
        plt.ylabel('Number of Customers')
        st.pyplot(plt.gcf())
        plt.close()

    # Gender-Based Segmentation Visualization
    if 'Gender' in df.columns:
        st.subheader('Gender-Based Segmentation Visualization')
        gender_segment_counts = df['Gender'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=gender_segment_counts.index, y=gender_segment_counts.values, palette='viridis')
        plt.title('Gender-Based Segmentation of Customers')
        plt.xlabel('Gender')
        plt.ylabel('Number of Customers')
        st.pyplot(plt.gcf())
        plt.close()

    # Location-Based Segmentation Visualization
    if 'Location' in df.columns:
        st.subheader('Location-Based Segmentation Visualization')
        location_segment_counts = df['Location'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=location_segment_counts.index, y=location_segment_counts.values, palette='coolwarm')
        plt.title('Location-Based Segmentation of Customers')
        plt.xlabel('Location')
        plt.ylabel('Number of Customers')
        st.pyplot(plt.gcf())
        plt.close()

    # Income Level Segmentation Visualization
    if 'Income Level' in df.columns:
        st.subheader('Income Level Segmentation Visualization')
        income_segment_counts = df['Income Level'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=income_segment_counts.index, y=income_segment_counts.values, palette='Spectral')
        plt.title('Income Level Segmentation of Customers')
        plt.xlabel('Income Level')
        plt.ylabel('Number of Customers')
        st.pyplot(plt.gcf())
        plt.close()

    # Product Category Preferences Segmentation Visualization (Pie Chart)
    if 'Product Category Preferences' in df.columns:
        st.subheader('Product Category Preferences Segmentation Visualization')
        product_category_counts = df['Product Category Preferences'].str.split(',', expand=True).stack().value_counts()
        plt.figure(figsize=(10, 8))
        plt.pie(product_category_counts.values, labels=product_category_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Set3'))
        plt.title('Product Category Preferences Segmentation')
        st.pyplot(plt.gcf())
        plt.close()

    # Customer Lifetime Value (CLV) Segmentation Visualization (Boxplot)
    if 'Customer Lifetime Value' in df.columns:
        st.subheader('Customer Lifetime Value (CLV) Segmentation Visualization')
        df['CLV Segment'] = df['Customer Lifetime Value'].apply(lambda x: 'High CLV' if x > df['Customer Lifetime Value'].mean() else 'Low CLV')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='CLV Segment', y='Customer Lifetime Value', data=df, palette='pastel')
        plt.title('Customer Lifetime Value (CLV) Segmentation')
        plt.xlabel('CLV Segment')
        plt.ylabel('Customer Lifetime Value')
        st.pyplot(plt.gcf())
        plt.close()

    # Combined Segmentation Visualization (Heatmap)
    if len(df.select_dtypes(include=['number']).columns) > 1:
        st.subheader('Combined Segmentation Visualization')
        segment_corr = df.select_dtypes(include=['number']).corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(segment_corr, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Between Customer Segments')
        st.pyplot(plt.gcf())
        plt.close()

# Function to display transactional data visualizations
def transactional_visualizations(df):
    st.title('Segmentation Visualization of Transactional Data')

    # Customer Segmentation
    def customer_segmentation(df):
        high_spend_threshold = df['Total Expenditure(till Date)'].median()
        df['High Spend Buyer'] = df['Total Expenditure(till Date)'] >= high_spend_threshold
        freq_buyer_threshold = df['Transaction ID'].nunique() / df['Customer ID'].nunique()
        df['Frequent Buyer'] = df.groupby('Customer ID')['Transaction ID'].transform('nunique') >= freq_buyer_threshold
        clv_threshold = df['Total Expenditure(till Date)'].quantile(0.75)
        df['High CLV'] = df['Total Expenditure(till Date)'] >= clv_threshold

    # Geolocation Segmentation
    def geolocation_segmentation(df):
        df['Max Product Sold'] = df.groupby('Shipping Address')['Quantity Purchased'].transform('sum')

    # Time-based Segmentation
    def time_based_segmentation(df):
        df['Season'] = pd.to_datetime(df['Transaction Date']).dt.month.map({
            12: 'Christmas Eve', 1: 'New Year', 10: 'Halloween'
        }).fillna('Other')
        last_purchase_date = df['Transaction Date'].max()
        df['Churn'] = (last_purchase_date - pd.to_datetime(df['Transaction Date'])).dt.days > 365

    # Apply segmentation functions
    customer_segmentation(df)
    geolocation_segmentation(df)
    time_based_segmentation(df)

    # High Spend vs Low Spend Buyer
    st.subheader('High Spend vs Low Spend Buyers')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='High Spend Buyer', palette='viridis')
    plt.title('High Spend vs Low Spend Buyers')
    plt.xlabel('High Spend Buyer')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Low Spend', 'High Spend'])
    st.pyplot(plt.gcf())
    plt.close()

    # Frequent Buyer vs Occasional Buyer
    st.subheader('Frequent Buyer vs Occasional Buyer')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Frequent Buyer', palette='viridis')
    plt.title('Frequent Buyer vs Occasional Buyer')
    plt.xlabel('Frequent Buyer')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Occasional Buyer', 'Frequent Buyer'])
    st.pyplot(plt.gcf())
    plt.close()

    # Customer Lifetime Value (CLV)
    st.subheader('Customer Lifetime Value Distribution')
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Total Expenditure(till Date)'], bins=30, kde=True, color='blue')
    plt.axvline(df['Total Expenditure(till Date)'].quantile(0.75), color='red', linestyle='--', label='CLV Threshold')
    plt.title('Customer Lifetime Value Distribution')
    plt.xlabel('Total Expenditure (till Date)')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

    # Max Product Sold by Geolocation
    st.subheader('Top 10 Locations by Max Products Sold')
    plt.figure(figsize=(14, 8))
    top_locations = df.groupby('Shipping Address')['Max Product Sold'].max().nlargest(10)
    sns.barplot(x=top_locations.index, y=top_locations.values, palette='viridis')
    plt.title('Top 10 Locations by Max Products Sold')
    plt.xlabel('Shipping Address')
    plt.ylabel('Total Quantity Purchased')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.close()

    # Seasonal Segmentation
    st.subheader('Seasonal Segmentation of Transactions')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Season', palette='viridis')
    plt.title('Seasonal Segmentation of Transactions')
    plt.xlabel('Season')
    plt.ylabel('Count')
    st.pyplot(plt.gcf())
    plt.close()

    # Churn Segmentation
    st.subheader('Churn Segmentation')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Churn', palette='viridis')
    plt.title('Churn Segmentation')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Not Churned', 'Churned'])
    st.pyplot(plt.gcf())
    plt.close()

# Streamlit app layout
