from __init__ import *

def upload_data():
    st.markdown("""
    ## Data Insight & Modeling Assistant:
    Welcome to the Data Insight & Modeling Assistant! Here you can upload your own dataset or choose one from the dropdown menu.
    """)

    # Dataset options for dropdown
    dataset_options = {
        'Select a Dataset': None,
        'IRIS Dataset (Clustering & Feature Selection)': 'Data/IRIS.csv',
        'Churn Dataset (Classification Analysis)': 'Data/churn.csv',
        'Decision Tree Classifier Dataset (Will Buy)': 'Data/DTClassiferWillBuy.csv',
        'Decision Tree Regression Dataset (Loan Amount)': 'Data/DTRegressionLoan.csv',
        'Park Data (Time Series ARIMA Analysis)': 'Data/ParkData_5years.csv',
        'Airbnb Dataset (Price Regression & Causality)': 'Data/df_selected1.csv',
        'Uncleaned Airbnb File (Data Cleaning & Exploration)': 'Data/listing.csv'
    }

    # Dropdown for dataset selection
    selected_dataset = st.selectbox('Want to explore?  We have these datasets ready to preload!', list(dataset_options.keys()))

    if selected_dataset != 'Select a Dataset':
        dataset_path = dataset_options[selected_dataset]
        st.session_state.data = pd.read_csv(dataset_path)
        st.write(f"{selected_dataset.split('(')[0].strip()} loaded:")
        st.write(st.session_state.data.head())

    # Option to clear the preloaded or uploaded file
    if st.button('Clear Data'):
        st.session_state.data = None
        st.write("Data cleared. You can now upload your own dataset or choose another from the dropdown.")

    # Step 1: Upload CSV
    uploaded_file = st.file_uploader("Or upload your CSV file", type="csv")

    if uploaded_file:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(st.session_state.data.head())

    # Provide a button for users to download the dataframe
    if 'data' in st.session_state and st.session_state.data is not None:
        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(st.session_state.data, 'your_data.csv', 'Click here to download the data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)