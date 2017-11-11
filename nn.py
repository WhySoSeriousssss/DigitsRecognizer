import numpy as np
import mnist

def relu(Z):
    A = np.maximum(Z, 0, Z)
    return A

    
def softmax(Z):
    T = np.exp(Z)
    A = T / np.sum(T, axis=0)
    return A

    
#initialize parameters
def initialize_parameters(layers_dim):
    para = dict()
    L = len(layers_dim)
    for i in range(1, L):
        para['W' + str(i)] = np.random.randn(layers_dim[i], layers_dim[i-1]) * 0.01
        para['b' + str(i)] = np.zeros((layers_dim[i], 1))
    return para

    
def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, Z)
    return Z, cache

    
def one_to_all_convertion(Y, num_channel):
    m = Y.shape[1]
    Y_all = np.zeros((num_channel, m))
    for i in range(num_channel):
        temp = np.ones((1, m))*i
        o = (temp == Y) * 1
        Y_all[i] = o
    return Y_all
    
    
def forward_propagation(X_Input, parameters):
    caches = list()
    A = X_Input
    L = len(parameters) // 2  #number of layers 3
    for i in range(1, L):
        A_prev = A
        Z, cache = linear_forward(A_prev, parameters['W' + str(i)], parameters['b' + str(i)])
        caches.append(cache)
        A = relu(Z)
#        print(A[:, 0])
    
    ZL, cache = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
#    print(ZL[:, 0])
    caches.append(cache)
    AL = softmax(ZL)
#    print(AL[:, 0])
    return AL, caches

    
def compute_cost_softmax(AL, Y):
    m = Y.shape[1]

    cost = -1/m * np.sum(np.multiply(np.log(AL), Y))
    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost

    
def linear_backward(dZ, A_prev, W):
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    return dW, db, dA_prev

    
def backward_relu(dA, Z):
    temp = (Z > 0)
    dZ = dA * temp
    return dZ

    
def backward_propagation(AL, Y, caches):
    grads = dict()
    L = len(caches) #L = 3
    dZL = AL - Y #softmax derivitive
    A_prev, W, Z = caches[L - 1]
    grads['dW'+str(L)], grads['db'+str(L)], grads['dA'+str(L-1)] = linear_backward(dZL, A_prev, W)
    
    for l in reversed(range(1, L)): #l = 2, 1
        A_prev, W, Z = caches[l - 1]
        dZ = backward_relu(grads['dA'+str(l)], Z)
        grads['dW'+str(l)], grads['db'+str(l)], grads['dA'+str(l-1)] = linear_backward(dZ, A_prev, W)
        
    return grads

    
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate * grads['dW'+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate * grads['db'+str(l)]
    return parameters


def compute_accuracy(X_test, Y_test, parameters):
    m = Y_test.shape[1]
    AL, _ = forward_propagation(X_test, parameters)
    Yp = AL.argmax(axis=0).reshape(1, -1)
    p = np.sum(Yp == Y_test) / m
    return p

########### start of the program ##############    

output_channel = 10
num_iteration = 2000

#load data and formalize them
X, Y = mnist.load_data("training")
XX, YY = mnist.load_data("testing")

X_train = X[0:30000, :,:]
Y_train = Y[:, 0:30000]
X_test = XX[0:5000, :,:]
Y_test = YY[:, 0:5000]

X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
Y_train_all = one_to_all_convertion(Y_train, output_channel)


#finalize layers size
img_size = X_train_flatten.shape[0] #28*28=784
hidden_layer_1 = 10
hidden_layer_2 = 5
layers = [img_size, hidden_layer_1, hidden_layer_2, output_channel]


#start training
parameters = initialize_parameters(layers)

for i in range(num_iteration):
    AL, caches = forward_propagation(X_train_flatten, parameters)

    cost = compute_cost_softmax(AL, Y_train_all)

    grads = backward_propagation(AL, Y_train_all, caches)

    parameters = update_parameters(parameters, grads, learning_rate = 0.008)

    if i % 100 == 0:
        print("cost after iteration% {}: {}".format(i, np.squeeze(cost)))


#compute the accuracy
print("training set accuracy = " + str(compute_accuracy(X_train_flatten, Y_train, parameters)))
print("test set accuracy = " + str(compute_accuracy(X_test_flatten, Y_test, parameters)))
