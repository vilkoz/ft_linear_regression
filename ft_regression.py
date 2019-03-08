#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from json import dumps
from sys import exit

def load_data():
    try:
        with open('data.csv') as f:
            data = f.read()
    except FileNotFoundError as e:
        print(e)
        exit(1)

    data = [x.split(',') for x in data.split('\n')]
    data = data[1:]
    x = [int(e[0]) for e in data if e[0] != '']
    y = [int(e[1]) for e in data if len(e) == 2]
    return x, y

def save_result(x, y, theta):
    data = {
            "x_mean": np.mean(x),
            "x_std": np.std(x),
            "y_mean": np.mean(y),
            "y_std": np.std(y),
            "theta": theta.tolist()
            }
    with open('coefs.json', 'w') as f:
        f.write(dumps(data))

def cost_function(x, y, theta):
    m = x.shape[0]
    return np.sum((x.T.dot(theta)-y)**2)/2/m

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def de_normalize(x, non_norm):
    return (x * np.std(non_norm)) + np.mean(non_norm)

def gradient_descent(x, y):
    x = normalize(x)
    y = normalize(y)
    theta = np.array([1, 0])
    learning_rate = 0.001
    iterations = 10000
    cost_history = []

    m = len(x)

    for _ in range(iterations):
        hypotesis = theta.dot(x)
        loss = hypotesis - y
        gradient = x.dot(loss) / m
        theta = theta - learning_rate * gradient

        cost = cost_function(x, y, theta)
        cost_history.append(cost)
    return theta, cost_history

def visualize_cost(cost_history):
    plt.plot([x for x in range(len(cost_history))], cost_history)
    plt.show()

def visualize_prediction(x, y, theta):
    x1 = normalize(x)
    y1 = x1.T.dot(theta)
    x1 = de_normalize(x1, x)
    y1 = de_normalize(y1, y)
    plt.plot(x.T[:,1], y, 'o')
    plt.plot(x1.T[:,1], y1, '-')
    plt.show()

def main():
    x, y = load_data()
    x, y = np.array(x), np.array(y)
    x = np.vstack((np.zeros((x.shape[0],)), x))
    theta, cost_history = gradient_descent(x, y)
    save_result(x, y, theta)
    visualize_cost(cost_history)
    visualize_prediction(x, y, theta)

if __name__ == "__main__":
    main()
