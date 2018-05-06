import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_decision_boundary(model, X, y):
    '''
    Plot the decision boundaries of a model trained on 2 dimensionnal data, as
    well as the scatter plot of the data X, using the target values in y for 
    the colors.
    ----------------------
    Input
            - model : Scikit-learn style estimator, already trained, must have
                    a .predict method
            - X : Data for scatter plot, must have 2 features.
            - y : Target classes for scatter plot colors
    '''
    # Set min and max values and give it some padding
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    # Predict the function value for the whole grid
    X_grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(X_grid)
    Z = Z.reshape(xx1.shape)
    # Plot the contour and training examples
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Spectral)
    plt.show()

def digit_plot(digit, ax=None):
    '''
    Plot a single digit.
    input : (1, 784) array
    '''
    try:
        digit_reshaped = digit.values.reshape(28,28)
    except:
        digit_reshaped = digit.reshape(28,28)
    if ax == None:
        fig, ax = plt.subplots()
    ax.imshow(digit_reshaped, cmap='gray_r')
    ax.axis('off')
    
def multidigits_plot(digits, size=None, shape=None, secure=True, titles=None):
    '''
    Plot n digits, max 100 digits if secure=True.
    
    Input : 
    - digits : dataframe of n digits or (n, 784) array.
    - size : integer, or list or tuple of 2 integers, to set size of figure.
    - shape : shape of the axes, then shape=size, except size is set to an integer.
    - secure : if set to True, will raise an error if digits has more than 100 rows.
    '''
    n = len(digits)
    if secure and n>100:
        raise ValueError('Too much digits to plot, make sure there is maximum 100 digits to plot, or set secure to False')
    
    # find the number of rows x, and columns y, for the axes of the plot
    if shape == None:
        for i in range(1, n):
            if (i*(i-1)) <= n:
                x = i
                y = i
                if x*y >= n:
                    break
                x = i
                y = i+1
                if x*y >= n:
                    break   
    else:
        x = shape[0]
        y = shape[1]
        if not isinstance(size, int):
            size = shape
    fig, ax = plt.subplots(x, y)
    ratio = x/y
    
    # set figure size
    if size==None:
        size = 8
        fig.set_size_inches(size/ratio, size*ratio)
    if isinstance(size, int):
        fig.set_size_inches(size/ratio, size*ratio)
    if isinstance(size, (list, tuple)):
        fig.set_size_inches(size[1], size[0])
    
    axes = ax.ravel()

    # plot the digits if digits is dataframe
    if isinstance(digits, pd.core.frame.DataFrame):
        for ax, index in zip(axes, digits.index):
            digit = digits.loc[index, :]
            digit_reshaped = digit.values.reshape(28,28)
            ax.imshow(digit_reshaped, cmap='gray_r')
    
    # plot the digits if digits is a 2D array
    else:
        for ax, digit in zip(axes, digits):
            digit_reshaped = digit.reshape(28,28)
            ax.imshow(digit_reshaped, cmap='gray_r')
    
    # hide axis
    for ax in axes:
        ax.axis('off')
        
    if titles is not None:
        for ax, title in zip(axes, titles):  
            ax.set_title(title)