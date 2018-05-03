'''
Created 02/05/2018

ANN with N layers and Softmax or sigmoid as the last layer.

Architecture of the NN : 
(Tanh or ReLU) x N-1 times + (Softmax or sigmoid) for output
'''
import numpy as np
import matplotlib.pyplot as plt
import time

def ReLU(Z):
    '''
    Compute the ReLU of the matrix Z
    '''
    relu = np.maximum(0, Z)
    
    return relu

def Sigmoid(Z):
    '''
    Compute the sigmoid of the matrix Z
    '''
    sigmoid = 1/(1+np.exp(-Z))
    
    return sigmoid

def Softmax(Z):
    '''
    Compute the Softmax of the matrix Z
    '''
    exp_Z = np.exp(Z)
    softmax = exp_Z/np.sum(exp_Z, axis=0)
    
    return softmax

def InitializeParameters(n_units_list):
    '''
    Initialize the parameters values W and b for each layers.
    --------
    Input  
            - n_units_list : list of number of units for each layers, input and
                            output included.
    Output
            - parameters : dictionnary of the parameters W and b
                           for each layers
    '''
    L = len(n_units_list) -1   # number of layers
    parameters = {}
    
    for l in range(1, L+1):
        n_l_prev = n_units_list[l-1]    # number of units in layer l-1
        n_l = n_units_list[l]           # number of units in layer l
        
        # initialize the parameters values randomly for W and 0 for b
        parameters['W' + str(l)] = np.random.randn(n_l, n_l_prev)*0.01
        parameters['b' + str(l)] = np.zeros((n_l, 1))
    
    return parameters

def ForwardProp(X, parameters, hidden_func, output_func):
    '''
    Compute the prediction matrix A3.
    --------
    Input
            - X : Matrix of input (n_x, m)
            - parameters : dictionnary of parameters W and b, for each layers
    Output
            - AL : The prediction matrix (n_y, m)
            - cache : Dictionnary of the A and Z, to use them during backprop
    '''
    L = len(parameters)//2
    cache = {}
    Al_prev = X
    for l in range(1,L):
        # get the parameters from the parameters dict
        Wl = parameters['W' + str(l)]
        bl = parameters['b' + str(l)]
        
        # compute forward propagation
        Zl = Wl.dot(Al_prev) + bl
                
        if hidden_func=='tanh':
            Al = np.tanh(Zl)
        if hidden_func=='relu':
            Al = ReLU(Zl)
        
        # write into the cache dict the Z and A
        cache['Z' + str(l)] = Zl
        cache['A' + str(l)] = Al
        
        # set Al_prev for next iter
        Al_prev = Al
        
    # compute forward prop for last layer
    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    ZL = WL.dot(Al_prev) + bL
    if output_func=='softmax':
        AL = Softmax(ZL)
    if output_func=='sigmoid':
        AL = Sigmoid(ZL)
        
    return AL, cache

def ComputeCost(Y, AL, output_func):
    '''
    Compute the cost function.
    --------
    Input
            - Y : Target matrix (n_y, m)
            - AL : Prediction matrix (n_y, m)
    Output
            - cost : the cost function computed for Y and AL
    '''
    n_y, m = Y.shape
    
    if output_func=='sigmoid':
        loss = - Y * np.log(AL) - (1-Y) * np.log(1 - AL)
        # sum the loss through the m examples
        cost = np.sum(loss)/m
        
    if output_func=='softmax':
        loss = - Y * np.log(AL)
        # sum the loss through the m examples
        cost = np.sum(loss)/(n_y*m)
        
    return cost


def BackProp(X, Y, AL, parameters, cache, output_func, hidden_func):
    '''
    Compute the gradients of the cost for the parameters W, b of each layers
    --------
    Input
            - X :
            - Y :
            - AL :
            - parameters : 
            - cache :
    Output
            - grads : dictionnary of the derivatives of the cost function
                      for each parameters
    '''
    m = X.shape[1]              # m = number of training examples
    L = len(parameters)//2       # L number of layer
    
    grads = {}
    
    # last layer
    # get dZL, depending of last layer activation fuction
    if output_func=='sigmoid':
        dZL = AL - Y
    if output_func=='softmax':
        dZL = AL - Y #(AL - 1) * Y
    
    # get AL_prev to compute the gradients
    AL_prev = cache['A'+str(L-1)]
    dWL = (1/m) * dZL.dot(AL_prev.T)
    dbL = (1/m) * np.sum(dZL, axis=1, keepdims=True)
    
    # write the gradients in grads dictionnary
    grads['dW'+str(L)] = dWL
    grads['db'+str(L)] = dbL
    
    # set dZl to dZL to be use as dZl_next for the first iter of the loop
    dZl = dZL
    
    # layer L-1 to 1
    for l in range(L-1,0,-1):
        # compute dAl
        Wl_next = parameters['W'+str(l+1)]
        dZl_next = dZl
        dAl = Wl_next.T.dot(dZl_next)
        
        # compute dZl
        Zl = cache['Z' + str(l)]
        if hidden_func=='tanh':
            dZl = (1 - np.tanh(Zl)**2) * dAl
        if hidden_func=='relu':
            dZl = (Zl > 0)*1
        
        # get Al_prev
        if l>1:
            Al_prev = cache['A'+str(l-1)]
        if l == 1:
            Al_prev = X
        
        # compute the gradients
        dWl = (1/m) * dZl.dot(Al_prev.T)
        dbl = (1/m) * np.sum(dZl, axis=1, keepdims=True)
        
        # write the gradients in grads dictionnary
        grads['dW'+str(l)] = dWl
        grads['db'+str(l)] = dbl
    
    return grads

def UpdateParameters(parameters, grads, learning_rate):
    '''
    Update the parameters by gradient descent
    ---------
    Input
            - parameters : dictionnary of parameters W, b of each layer
            - grads : dictionnary of gradient of the cost function
                      for each parameters W, b of each leayer
            - learning_rate : learning rate to use for updating the parameters
    Output
            - parameters : parameters updated after gradient descent
    '''
    L = len(parameters)//2           # L number of layer
    
    for l in range(1, L+1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    
    return parameters

def NN_model(X, Y, hidden_units, hidden_func='tanh', output_func='sigmoid', \
             epoch=10000, learning_rate=0.01, verbose=True, grad_check=False):
    '''
    Train a Neural Network of 3 layers (2 layers ReLU and 1 sigmoid for the output).
    ----------
    Input
            - X : input training dataset (m, n_x)
            - Y : target of the training dataset (m, 1)
            - layer_units : tuple of number of units for the 2 ReLU layers
            - epoch : number of iteration
            - learning_rate : learning rate for the gradient descent
            - verbose : if True, print cost function value every 100 epoch
    Output
            - parameters : dictionnary of the trained parameters W, b for each layers
    '''
    t0 = time.time()
    # transpose X and Y
    X = X.T
    Y = Y.T
    
    # get the number of features n_x and number of examples m of X
    n_x, m = X.shape
    # get the number of classes n_y
    n_y = Y.shape[0]
    
    n_units_list = [n_x] + hidden_units + [n_y]
    # initialize the parameters
    parameters = InitializeParameters(n_units_list)
    
    # initialize a list to plot the evolution of the cost function
    cost_list = []
    for i in range(epoch):        
        # compute the forward propagation
        AL, cache = ForwardProp(X, parameters, hidden_func, output_func)
        
        # compute the back propagation
        grads = BackProp(X, Y, AL, parameters, cache, output_func, hidden_func)
        
        # update the parameters
        parameters = UpdateParameters(parameters, grads, learning_rate)
        
        if  i%100 == 0:
            # compute the cost function
            cost = ComputeCost(Y, AL, output_func)
            cost_list.append(cost)
            
            if verbose and (i%1000 == 0):
                print('Cost function after epoch {} : {}'.format(i, cost))
        
        if grad_check:
            grad_diff = GradCheck(X, Y, parameters, grads, hidden_func, output_func, epsilon=1e-7)
            print('Gradient evaluation :', grad_diff)
            if i > 10:
                print('OK if less than 1e-5')
                break
                
    print('Cost function after epoch {} : {}'.format(epoch, cost))
    print('Time : %.3f s' % (time.time()-t0))
    
    # print the cost function for each iterations
    plt.plot(cost_list)
    plt.title('Cost function')
    plt.xlabel('Number of iterations, by hundreds')
    plt.ylabel('Cost Function')
    
    return parameters

def MakePrediction(X, parameters, hidden_func, output_func):
    '''
    Make prediction of the data X
    ---------
    Input
            - X : Input data (m, n_x)
            - parameters : parameters W, b of each layers of the NN model
    Output
            - Y_pred : Predicted labels for X (m, n_y)
    '''
    X = X.T
    A3, _ = ForwardProp(X, parameters, hidden_func, output_func)
    
    if output_func=='softmax':
        
        prediction = np.argmax(A3, axis=0)
        Y_pred = np.zeros(A3.shape)
        
        for i in range(len(prediction)):
            Y_pred[prediction[i], i] = 1
            
    if output_func=='sigmoid':
        Y_pred = (A3 >0.5)*1
        
    Y_pred = Y_pred.T # transpose the prediction to get the usual form (m, n_y)
    
    return Y_pred

def GradCheck(X, Y, parameters, grads, hidden_func, output_func, epsilon=1e-7):
    '''
    Piece of crap, ugly and not working , do not use.
    
    Check if the gradient computed by backprop is correct
    '''
    grads_approx = {}
    for key in parameters:
        grads_approx[key] = np.zeros(parameters[key].shape)
        n, m = parameters[key].shape
        for i in range(n):
            for j in range(m):
                parameters_plus = parameters.copy()
                parameters_minus = parameters.copy()
                
                parameters_plus[key] = parameters[key].copy()
                parameters_minus[key] = parameters[key].copy()
                
                parameters_plus[key][i, j] +=  epsilon
                parameters_minus[key][i, j] -=  epsilon
                
                AL_plus, _ = ForwardProp(X, parameters_plus, hidden_func, output_func)
                cost_plus = ComputeCost(Y, AL_plus, output_func)
                
                AL_minus, _ = ForwardProp(X, parameters_minus, hidden_func, output_func)
                cost_minus = ComputeCost(Y, AL_minus, output_func)
                
                grads_approx[key][i, j] = (cost_plus - cost_minus)/(2*epsilon)
                
    grads_approx_array = np.array([])
    grads_array = np.array([])
    L = len(parameters)//2
    for l in range(1, L+1):
        grads_approx_array = np.append(grads_approx_array, grads_approx['W'+str(l)].flatten())
        grads_approx_array = np.append(grads_approx_array, grads_approx['b'+str(l)].flatten())
        
        grads_array = np.append(grads_array, grads['dW'+str(l)].flatten())
        grads_array = np.append(grads_array, grads['db'+str(l)].flatten())
    
    grads_approx_array_norm = np.linalg.norm(grads_approx_array)
    grads_array_norm = np.linalg.norm(grads_array)
    grad_diff = np.linalg.norm(grads_approx_array - grads_array)/(grads_approx_array_norm + grads_array_norm)
    
    return grad_diff


