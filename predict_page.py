import streamlit as st
import numpy as np
import pickle
from explore_page import compute_cost, compute_gradient, gradient_descent

# Load the pickled model
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

predicted_loaded = data["model"]
population = data["Population"]






def show_predict_page():
    st.title("Restaurant Profit Prediction App")

    #population = st.slider("Population", 0, 100, 50) 
    population = st.number_input("type in the population", 0, 100, 50)
    x_train = data['Population']
   # y_train = data['Profit']
    #initial_w = 0
    #initial_b = 0

    # some gradient descent values
    #iterations = 150
    #alpha = 0.01


    #w,b,_,_ = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost,
      #                     compute_gradient, alpha, iterations)
    w = 1.17
    b = -3.63

    ok = st.button("Predict Profit")
    if ok:
        m = x_train.shape[0]
        predicted = np.zeros(m)
        for i in range(m):
            predicted[i] = w * x_train[i] + b

        profit =  population * w +b 
        st.subheader(f"The profit of the restaurant is ${profit*10000:.2f} dollars")
            #st.success(f"The profit of the restaurant is {profit:.2f} dollars")
            #st.write("The predicted profit is", predicted)
            #predict_1 = 3.5 * w + b
