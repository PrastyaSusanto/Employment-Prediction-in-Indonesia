import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

title = 'Application for predicting employed and unemployed individuals in Indonesia👷‍♀️👷🏢'
subtitle = 'Predict the number of employed and unemployed individuals in Indonesia using Machine Learning👷‍♀️👷🏢 '

def main():
    st.set_page_config(layout="centered", page_icon='🏢💻', page_title='Lets Predicting the number of workers and employed population')
    st.title(title)
    st.write(subtitle)
    st.write("For more information about this project, check here: [GitHub Repo](https://github.com/PrastyaSusanto/Commuter-Prediction-App/tree/main)")

    form = st.form("Data Input")
    options = ['Employed', 'Unemployed']
    selected_options = form.multiselect('Employed/Unemployed', options, default=options)
    start_date = form.date_input('Start Date')
    end_date = form.date_input('End Date')

    submit = form.form_submit_button("Predict")  # Add a submit button

    if submit:
        data = {
            'Kode Tipe': [1 if option == 'Unemployed' else 0 for option in selected_options],
            'Tanggal Referensi': pd.date_range(start=start_date, end=end_date).to_list()
        }
        data = pd.DataFrame(data)

        # Convert Tanggal column to datetime and calculate the difference from the reference date
        data['Tanggal Referensi'] = (pd.to_datetime(data['Tanggal Referensi']) - pd.to_datetime('2011-02-01')).dt.days

        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Make prediction using the loaded model
        predictions = model.predict(data)

        # Create a DataFrame to store the results
        results = pd.DataFrame({
            'Date': pd.date_range(start=start_date, end=end_date),
            'Employed': predictions[:, 0],  # Predicted values for 'Employed'
            'Unemployed': predictions[:, 1]  # Predicted values for 'Unemployed'
        })

        # Format the predicted values as integers
        results['Employed'] = results['Employed'].astype(int)
        results['Unemployed'] = results['Unemployed'].astype(int)

        # Visualize the results using matplotlib
        plt.style.use('dark_background') 
        plt.plot(results['Date'], results['Employed'], label='Employed', color='royalblue')
        plt.plot(results['Date'], results['Unemployed'], label='Unemployed', color='salmon')
        plt.xlabel('Date')
        plt.ylabel('Predicted Number')
        plt.xticks(rotation=90)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend()
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))  # Set maximum number of x-axis ticks
        plt.title('Predicted Amount of Employed and Unemployed over Time')
        
        st.pyplot(plt)

        # Format the Date column in the results DataFrame
        results['Date'] = results['Date'].dt.strftime('%d-%m-%Y')

        st.dataframe(results)
    

if __name__ == '__main__':
    main()
