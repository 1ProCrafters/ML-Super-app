from __init__ import *

def create_model():
    
    st.title("Create Model")

    sub_page = st.sidebar.radio(
        "Choose a model to create:",
        ["LLM", "", "", "",]
    )
    if sub_page == "LLM":
        create_LLM()


def create_LLM():
    pass