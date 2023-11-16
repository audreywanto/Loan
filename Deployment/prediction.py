import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load dump files
with open('dt_model.pkl', 'rb') as file_1:
    dt_model = pickle.load(file_1)

with open('model_scalar.pkl', 'rb') as file_2:
    model_scalar = pickle.load(file_2)
    
with open('model_encoder.pkl', 'rb') as file_3:
    model_encoder = pickle.load(file_3)
    
with open('num_col.txt', 'r') as file_4: 
    num_col = json.load(file_4)

with open('cat_col.txt', 'r') as file_5: 
    cat_col = json.load(file_5)
    
def run():
    # Membuat form
    with st.form(key='form parameters'):
        loan_id = st.text_input('Loan ID', value='', help = 'Integer Value Only')
        no_of_dependents = st.selectbox('Number of Dependents', (1, 2, 3, 4, 5, 6, 7, 8, 9), help='Number of People in the main family, if more than 9, select 9')
        education = st.selectbox('Education', ('Graduate', 'Not Graduate'), help='Education status')
        self_employed = st.selectbox('Employment Status', ('Yes', 'No'), help='Yes for Employed, and No for Not Employed')
        
        st.markdown('---')
        
        income_annum = st.text_input('Annual Income', value='', help = 'Integer Value Only')
        loan_amount = st.text_input('Loan Amount', value='', help = 'Integer Value Only')
        loan_term = st.slider('Loan Term in Years', 0, 50, 0)
        cibil_score = st.slider('Credit Score', 300, 900, 600)
        residential_assets_value = st.text_input('Residential Assets Value', value='', help = 'Integer Value Only')
        commercial_assets_value = st.text_input('Commercial Assets Value', value='', help = 'Integer Value Only')
        luxury_assets_value = st.text_input('Luxury Assets Value', value='', help = 'Integer Value Only')
        bank_asset_value = st.text_input('Bank Asset Value', value='', help = 'Integer Value Only')
        total_assets_value = st.text_input('Total Assets Value', value='', help = 'Total Value from all Assets combined, Integer Value Only' )
        condition = st.selectbox('Is your total assets value higher than the loan amount?', ('unfulfilled', 'fulfilled'), help='unfulfilled for No, fulfilled for Yes')

        st.markdown('---')
        
        submitted = st.form_submit_button('Predict')
        
    data_inf = {
    'loan_id':loan_id, 
    'no_of_dependents':no_of_dependents, 
    'education':education, 
    'self_employed':self_employed,
    'income_annum':income_annum, 
    'loan_amount':loan_amount, 
    'loan_term':loan_term, 
    'cibil_score':cibil_score,
    'residential_assets_value':residential_assets_value, 
    'commercial_assets_value':commercial_assets_value,
    'luxury_assets_value':luxury_assets_value, 
    'bank_asset_value':bank_asset_value, 
    'total_assets_value':total_assets_value, 
    'condition':condition}
    
    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)
    
    if submitted:
        # Split between num col and cat col
        data_inf_num = data_inf[num_col]
        data_inf_cat = data_inf[cat_col]
        
        # Feature scaling and encoding
        data_inf_num_scaled = model_scalar.transform(data_inf_num)
        data_inf_cat_encoded = model_encoder.transform(data_inf_cat)
        
        # Make cat_encoded into a dataframe so it can be concatenated
        data_inf_cat_encoded_df = pd.DataFrame(data_inf_cat_encoded.toarray(), columns=model_encoder.get_feature_names_out(data_inf_cat.columns))
        
        # Concat data inference scaled and encoded
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded_df], axis=1)
        
        # Predict using Linear Regression
        y_pred_inf = dt_model.predict(data_inf_final)
        
        # Print result of the prediction
        if int(y_pred_inf) == 0:
            st.write('The prediction for `loan_status` is: 0 (Rejected)')
        elif int(y_pred_inf) == 1:
            st.write('The prediction for `loan_status` is: 1 (Approved)')
        
if __name__== '__main__':
    run()