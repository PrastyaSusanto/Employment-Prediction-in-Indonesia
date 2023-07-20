import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import PolynomialFeatures

title = 'Application for predicting employed and unemployed individuals in Indonesia👷‍♀️👷🏢'
subtitle = 'Predict the number of employed and unemployed individuals in Indonesia using Machine Learning👷‍♀️👷🏢 '

def main():
    st.set_page_config(layout="centered", page_icon='🏢💻', page_title='Lets Predicting the number of workers and employed population')
    st.title(title)
    st.write(subtitle)
    st.write("For more information about this project, check here: [GitHub Repo](https://github.com/PrastyaSusanto/Commuter-Prediction-App/tree/main)")

    form = st.form("Data Input")
    Option = form.selectbox('Employed/Unemployed', ['Employed', 'Unemployed'])
    start_date = form.date_input('Start Date')
    end_date = form.date_input('End Date')

    submit = form.form_submit_button("Predict")  # Add a submit button

    if submit:
        data = {
            'Kode Tipe': Option,
            'Tanggal Referensi': pd.date_range(start=start_date, end=end_date).to_list()
        }
        data = pd.DataFrame(data)

        data['Kode Tipe'] = data['Kode Wilayah'].replace({'Employed': 0, 'Unemployed': 1})

        # Convert Tanggal column to datetime and calculate the difference from the reference date
        data['Tanggal Referensi'] = (pd.to_datetime(data['Tanggal Referensi']) - pd.to_datetime('2011-02-01')).dt.days

        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction using the loaded model
        predictions = model.predict(data)

        # Create a DataFrame to store the results
        results = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date), 'Predicted Number': predictions})

        # Format the Date column in the results DataFrame
        results['Date'] = results['Date'].dt.strftime('%d-%m-%Y')

        # Visualize the results using matplotlib
        plt.style.use('dark_background') 
        plt.plot(results['Date'], results['Predicted Number'], color='royalblue')
        plt.xlabel('Date')
        plt.ylabel('Predicted Number')
        plt.xticks(rotation=90)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title('Predicted Amount of '+Option+" over Time")
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # Set maximum number of x-axis ticks
        
        st.pyplot(plt)

        # Show the table with three columns: Date, Employed, and Unemployed
        st.dataframe(results)
    

if __name__ == '__main__':
    main()
