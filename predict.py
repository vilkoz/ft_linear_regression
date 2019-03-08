#!/usr/bin/env python3
import numpy as np
from json import loads
from sys import exit
import ft_regression
import matplotlib.pyplot as plt

def load_coefs():
    try:
        with open('coefs.json') as f:
            data = f.read()
    except FileNotFoundError as e:
        print('File "coefs.json" not found, please run training program')
        exit(1)
    data = loads(data)
    return data['theta'], data['x_mean'], data['x_std'], data['y_mean'], data['y_std']

def main():
    theta, x_mean, x_std, y_mean, y_std = [np.array(x) for x in load_coefs()]
    while True:
        try:
            mileage_str = input('enter mileage: ')
            mileage = int(mileage_str)
            break
        except ValueError as e:
            print('Error: {}'.format(e))
        except EOFError:
            print('\nError: {}'.format('EOFError'))
    x = np.zeros((2,))
    x[1] = mileage
    x = (x - x_mean) / x_std
    y1 = x.T.dot(theta)
    y1 = (y1 * y_std) + y_mean
    print('Estimate: {}'.format(y1))

if __name__ == "__main__":
    main()


    
