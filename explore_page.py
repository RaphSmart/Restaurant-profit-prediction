import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import math


# Compute the cost function
def compute_cost(x, y, w, b):
    m = x.shape[0] # m is the number of training examples
    total_cost = 0
    cost_sum = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
        total_cost = (1 / (2 * m)) * cost_sum
        
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0] # number of training examples
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        # Update parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost J at each iteration
        if i<100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
            
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f} ")
            
    return w, b, J_history, w_history


def load_data():
    header = ['Population', 'Profit']
    data = pd.read_csv('dataset/restaurant.csv', sep = ';', names=header)
    x_train = data['Population']
    y_train = data['Profit']
    # Compute cost with some initial values for parameters w, b
    initial_w = 2
    initial_b = 1

    cost = compute_cost(x_train, y_train, initial_w, initial_b)

    initial_w = 0
    initial_b = 0

    tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)

    # Compute and display cost and gradient with non-zero w and b
    test_w = 0.2
    test_b = 0.2
    tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

    initial_w = 0
    initial_b = 0

    # some gradient descent values
    iterations = 150
    alpha = 0.01


    w,b,_,_ = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost,
                            compute_gradient, alpha, iterations)


data = load_data()






# display page
def show_explore_page():
   st.title("Restaurant profit prediction")
