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
    options = ['Employed', 'Unemployed']
    selected_options = form.multiselect('Employed/Unemployed', options, default=options)
    start_date = form.date_input('Start Date')
    end_date = form.date_input('End Date')

    submit = form.form_submit_button("Predict")  # Add a submit button

    if submit:
        # Create a list to store the selected 'Option' for each date
        option_list = []
        for date in pd.date_range(start=start_date, end=end_date):
            option_list.extend(selected_options)

        data = {
            'Tanggal Referensi': pd.date_range(start=start_date, end=end_date).repeat(len(selected_options)),
            'Option': [0 if option == 'Employed' else 1 for option in option_list]
        }
        data = pd.DataFrame(data)

        # Convert Tanggal column to datetime and calculate the difference from the reference date
        data['Tanggal Referensi'] = (data['Tanggal Referensi'] - pd.to_datetime('2011-02-01')).dt.days

        # Load the model from the pickle file
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Use PolynomialFeatures with degree 2 to transform input data
        poly = PolynomialFeatures(degree=2)
        data_poly = poly.fit_transform(data[['Tanggal Referensi', 'Option']])

        # Make prediction using the loaded model and transformed data_poly
        predictions = model.predict(data_poly)

        # Create a DataFrame to store the results
        results = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date).repeat(len(selected_options)), 'Option': option_list, 'Predicted Number': predictions})

        # Format the predicted passenger values as integers
        results['Predicted Number'] = results['Predicted Number'].astype(int)

        # Pivot the DataFrame to create separate columns for Employed and Unemployed
        pivot_results = results.pivot(index='Date', columns='Option', values='Predicted Number')
        pivot_results.columns = ['Employed', 'Unemployed']

        # Visualize the results using two line charts
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for i, option in enumerate(selected_options):
            axes[i].plot(pivot_results.index, pivot_results[option], label=option)
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Predicted Number')
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].set_title('Predicted Amount of {} over Time'.format(option))
            axes[i].xaxis.set_major_locator(MaxNLocator(nbins=10))  # Set maximum number of x-axis ticks
            axes[i].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Format the Date column in the results DataFrame
        results['Date'] = results['Date'].dt.strftime('%d-%m-%Y')

        # Show the table with three columns: Date, Employed, and Unemployed
        st.dataframe(pivot_results)

if __name__ == '__main__':
    main()
