import pandas as pd
from sklearn.cluster import KMeans
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
df = pd.read_csv(r"C:\Users\rshic\Downloads\Mall_Customers.csv")

# Exclude non-numeric columns for correlation matrix
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Save the model
model_filename = 'kmeans_model.joblib'
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])
joblib.dump(kmeans, model_filename)

# Streamlit App
st.title("Customer Segmentation App")

# Sidebar for user input
st.sidebar.header("Choose Data Exploration Option")
exploration_option = st.sidebar.radio("Select Option", ["EDA", "Clustering"])

if exploration_option == "EDA":
    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")

    # Display first few rows of the dataset
    st.write("First few rows of the dataset:")
    st.write(df.head())

    # Summary statistics
    st.write("Summary statistics:")
    st.write(df.describe())

    # Data types and missing values
    st.write("Data types and missing values:")
    st.write(df.info())

    # Set seaborn style to whitegrid to remove warnings
    sns.set(style="whitegrid")

    # Pair plot for numerical features with try-except block to handle warnings
    try:
        st.write("Pair Plot of Numerical Features:")
        sns.pairplot(df, hue='Gender')
        st.pyplot()
    except Exception as e:
        st.write(f"Error creating pair plot: {e}")

    # Distribution of Age
    st.write("Distribution of Age:")
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], bins=20, kde=True)
    st.pyplot()

    # Box plot for Annual Income and Spending Score by Gender
    st.write("Box plot for Annual Income and Spending Score by Gender:")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(x='Gender', y='Annual Income (k$)', data=df, ax=axes[0])
    axes[0].set_title('Annual Income by Gender')

    sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df, ax=axes[1])
    axes[1].set_title('Spending Score by Gender')

    st.pyplot()

    # Correlation matrix
    st.write("Correlation Matrix:")
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()

else:
    # Clustering Analysis
    st.subheader("Customer Clustering Analysis")

    # Display the clusters on a scatter plot
    st.write("Scatter plot of Annual Income vs. Spending Score with Clusters:")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, hue='Cluster', palette='viridis')
    st.pyplot()

    # Save the plot
    plt.savefig('clusters_plot.png')

    # Show the saved plot
    st.image('clusters_plot.png')

# Display the app
st.set_option('deprecation.showPyplotGlobalUse', False)
