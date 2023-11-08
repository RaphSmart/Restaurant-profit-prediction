import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

page = st.sidebar.selectbox("Choose a page", ["Prediction", "Exploration"])

#st.title("Restaurant Profit Prediction App")
#st.write("This is a simple Restaurant Profit Prediction App using Streamlit")
# Restaurant Profit Prediction App

if page == "Prediction":
    show_predict_page()
else:
     show_explore_page()

