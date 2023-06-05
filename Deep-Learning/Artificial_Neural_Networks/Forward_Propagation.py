'''
Artificial Neural Networks - Forward Propagation
    1. Initalize a network
    2. Compute Weighted Sum at each node
    3. Compure node activation
    4. Acces your flask app via webpage anywhere using a custom link
'''

import numpy as np
import sys  # to access the system
import cv2

img = cv2.imread(
    'Deep-Learning/Artificial_Neural_Networks/neural_network_example.png',
    cv2.IMREAD_ANYCOLOR)

print('--------- Forward Propagation ---------')

weights = np.around(np.random.uniform(size=6), decimals=2)
biases  = np.around(np.random.uniform(size=3), decimals=2)

print('Weights: ', weights)
print('Biases:  ', biases)

x_1 = 0.5  # Input 1
x_2 = 0.85 # Input 2
print('x1 is {} and x2 is {}'.format(x_1, x_2))

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The z1,1 of the first node in the hidden layer is {}'.format(
    np.round(z_11, decimals=2)))

z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
print('The z1,2 of the second node in the hidden layer is {}'.format(
    np.round(z_12, decimals=2)))

a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(
    np.round(a_11, decimals=2)))

a_12 = 1.0 / (1.0 + np.exp(-z_12))
print('The activation of the first node in the hidden layer is {}'.format(
    np.round(a_12, decimals=2)))

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print(
    'The weighted su, of the inputs at the node in the output layer (z2) is {}'.
    format(np.round(z_2, decimals=2)))

a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0,85 is {}'.
      format(np.round(a_2, decimals=2)))

while True:
    cv2.imshow('Neural Network Example', img)
    cv2.waitKey(0)
    break
    sys.exit()
    

cv2.destroyAllWindows()

print('--------- Automatic Forward Propagation ---------')
n = 2  # number of inputs
num_hidden_layers = 2 
m = [2, 2] # number of nodes in each hidden layer
num_nodes_output = 1

num_nodes_previous = n # number of nodes in the previous layer
network = {} # initalize network as an empty dictionary

# loop through each layer and randomly initialize the weights and biases 
# associated with each node, notice how we are adding 1 to the number of hidden
# layers in order to include the output layer
for layer in range(num_hidden_layers + 1):

    # determine name of layer
    if layer == num_hidden_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output
    else:
        layer_name = 'layer_{}'.format(layer + 1)
        num_nodes = m[layer]

    # initialize weights and biases associated with each node in the 
    # current layer
    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = 'node_{}'.format(node + 1)
        network[layer_name][node_name] = {
            'weights' : np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias' : np.around(np.random.uniform(size=1), decimals=2)
        }
    
    num_nodes_previous = num_nodes

print('Shape of the network:\n', network)

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {}
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes

    return network # return the network