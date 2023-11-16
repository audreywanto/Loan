import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose a page:',('EDA', 'Predict Loan Status'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()