from __init__ import *

def sanitize_string(s):
    """Removes non-UTF-8 characters from a string."""
    return s.encode('utf-8', 'ignore').decode('utf-8')

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def save_data():
    """
    Provides a button to save the current dataframe to CSV.
    """
    # Check if there's data in the session state
    if 'data' in st.session_state and st.session_state.data is not None:
        # Provide a button for users to download the dataframe
        if st.button('Download Dataframe as CSV'):
            tmp_download_link = download_link(st.session_state.data, 'your_data.csv', 'Click here to download the data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
    else:
        st.warning("No data available to save. Please upload data first.")

def main_updated():
    # Sidebar for primary task selection
    primary_task = st.sidebar.radio(
    "Choose a primary task:",
    ["Data Upload", "Feature Engineering", "Explore the Data", "Regression Analysis", "Extensive Data Analysis", "Time Series Analysis", "Causality Analysis", "Decision Tree Analysis", "Save", "Create Model"]
    )

    if primary_task == "Data Upload":
        upload_data()
    elif primary_task == "Feature Engineering":
        feature_engineering()
    elif primary_task == "Explore the Data":
        explore_data()
    elif primary_task == "Regression Analysis":
        evaluate_model_page()
    elif primary_task == "Extensive Data Analysis":
        advanced_data_analysis()
    elif primary_task == "Time Series Analysis":
        time_series_analysis()
    elif primary_task == "Save":
        save_data()
    elif primary_task == "Causality Analysis":
        causality_page()
    elif primary_task == "Decision Tree Analysis":
        decision_tree_page.decision_tree_page()
    elif primary_task == "Create Model":
        create_model_page.create_model()

if __name__ == "__main__":
    main_updated()