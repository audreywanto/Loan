import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title='Loan Approval Dataset - EDA',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Creating title
    st.title('Loan Status Approval Prediction')
    
    # Creating Sub Header
    st.subheader('EDA for Loan Application Dataset')
    
    # Menambahkan gambar
    st.image('https://lp2m.uma.ac.id/wp-content/uploads/2022/09/loan.jpg', 
             caption='Loan Thumbnail')
    
    # Menambahkan deskripsi
    st.write('Page written and coded by Audrey Wanto')
    st.write('Batch: BSD 002')
    st.write('Milestone 2 Project')
    
    st.write('---')
    
    # Magic syntax
    '''
    Within this page, there will be simple Exploratory
    Data Analysis from the Loan Application Dataset.
    '''
    
    # Show Dataframe
    df_1 = pd.read_csv('loan_approval_dataset.csv')
    st.dataframe(df_1)
    
    # Scatterplot between Limit Balance and Education Level, using marital status as hue
    st.write('#### Scatterplot of `cibil_score` and `loan_amount`, differentiated by `loan_status` ')
    fig = plt.figure(figsize=(15,5))
    sns.scatterplot(x=' cibil_score', y=' loan_amount', hue=' loan_status', data=df_1)
    st.pyplot(fig)
    
    # Create Histograms
    st.write('#### Histogram of income_annum and loan_amount')
    # Create canvas
    fig = plt.figure(figsize=(15,5))

    # Plot 1 for 'price' histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df_1[' income_annum'], kde=True, bins=30)

    # Plot 2 for 'distance' histogram
    plt.subplot(1, 2, 2)
    sns.histplot(df_1[' loan_amount'], kde=True, bins=30)

    # Show all plot
    st.pyplot(fig)
    
    # Create Histograms
    st.write('#### Histogram of loan_term and cibil_score')

    # Create canvas
    fig = plt.figure(figsize=(15,5))

    # Plot 1 for 'price' histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df_1[' loan_term'], kde=True, bins=30)

    # Plot 2 for 'distance' histogram
    plt.subplot(1, 2, 2)
    sns.histplot(df_1[' cibil_score'], kde=True, bins=30)

    # Show all plot
    st.pyplot(fig)
    
    # Pie chart for 'loan_status'
    st.write('#### Pie Chart of loan_status')
    fig = plt.figure(figsize=(15,5))
    loan_total = df_1[' loan_status'].value_counts().values.sum()
    def fmt(x):
        return '{:.1f}%\n{:.0f}'.format(x, loan_total*x/100)
    plt.pie(df_1[' loan_status'].value_counts().values, labels=df_1[' loan_status'].value_counts().index, autopct=fmt)
    st.pyplot(fig)
    
    # Pie chart for 'self_employed'
    st.write('#### Pie Chart of self_employed')
    fig = plt.figure(figsize=(15,5))
    loan_total = df_1[' self_employed'].value_counts().values.sum()
    def fmt(x):
        return '{:.1f}%\n{:.0f}'.format(x, loan_total*x/100)
    plt.pie(df_1[' self_employed'].value_counts().values, labels=df_1[' self_employed'].value_counts().index, autopct=fmt)
    st.pyplot(fig)

if __name__== '__main__':
    run()