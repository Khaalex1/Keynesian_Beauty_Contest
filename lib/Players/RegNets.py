from Keynesian_Beauty_Contest.lib.Players.crowd import *

def sigmoid(z):
    """
    sigmoid function
    :param z:
    :return: sigmoid evaluation of z
    """
    return (1 / (1 + np.exp(-z)))

def composed_sigmoid(z):
    """
    Investigated output activation in the KBC : Int(100*sigmoid)
    :param z:
    :return: evaluation of z
    """
    return np.asarray(100*sigmoid(z), dtype='int')


def tanh(z):
    """
    Tanh activation
    :param z:
    :return: eval of z
    """
    return ((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))) / 2


def relu(z):
    """
    ReLu activation function
    :param z:
    :return: eval of z
    """
    return np.maximum(0, z)


def forward(W, X, layers = [2, 10, 10, 1], activation = [None, relu, relu, composed_sigmoid]):
    """
    Forward pass function
    :param W: 1-D flattened array of weights and then biases
    :param X: input values of network. array (1, layers[0])
    :param layers: list containing the nb of nodes of each layer, including input and output ones
    :param activation: list of activation function used at the end of each layer concerned
    :return: forward evaluation of X. array (1, layers[-1])
    """
    Weights = len(layers)*[None]
    Bias = len(layers)*[None]
    weight_shapes = len(layers)*[None]
    bias_shapes = len(layers)*[None]
    for i in range(1, len(layers)):
        weight_shapes[i] = layers[i-1]*layers[i]
        bias_shapes[i] = layers[i]
    # number of weights (without biases)
    sum_weight_shapes = np.sum(weight_shapes[1:])
    w_cursor = 0
    #biases are at the end of the vector W
    b_cursor = sum_weight_shapes
    # reformating W into actual neural architecture with layers
    for j in range(1, len(layers)):
        Weights[j] = W[w_cursor:(w_cursor + weight_shapes[j])].reshape((layers[j], layers[j-1]))
        Bias[j] = W[b_cursor : (b_cursor + bias_shapes[j])].reshape((bias_shapes[j], 1))
        w_cursor += weight_shapes[j]
        b_cursor += bias_shapes[j]
    Z = len(layers)*[None]
    Out = [X] + (len(layers) - 1)*[None]
    #forward pass. Out[0] = layer[0] = input
    for k in range(1, len(layers)):
        # Forward pass for the layer 'k'
        Z[k] = Weights[k] @ Out[k - 1] + Bias[k]
        Out[k] = activation[k](Z[k])
    return Out[-1]


def RegNet(config, agent_memory, layers = [1, 10, 10, 1]):
    """
    agent using the forward pass as descision function (pseudo-correct signature)
    :param config: 1-S array of flattened weights of the neural network
    :param agent_memory: list of previous memories (prev. avgs)
    :param layers: list of layer nodes
    :return: response to memory via forward pass or default play
    """
    #default play if not enough memory (input size = layers[0])
    if len(agent_memory)<layers[0]:
        play = 50
    else:
        # only two last memories to launch response
        X = np.array(agent_memory[-layers[0]:]).reshape((-1,1))
        play = forward(config, X, layers=layers)[0,0]
    return play


def memory_RegNet(layers):
    """
    Return RegNet Agent function with memory embedded and correct signature
    :param layers: list of layer nodes
    :return: RegNet agent/function, without having to specify layers
    """
    def evaluate(config, agent_memory):
        return RegNet(config, agent_memory, layers)
    return evaluate

def HybridNet(config, agent_memory):
    """
    Combines play of RegNet1 in round 1, and RegNet2 for next rounds
    :param config: flattened weight vector of RegNet1 AND RegNet2
    :param agent_memory: list of previous memories (prev. avgs)
    :return: response to memory via forward pass or default play
    """
    config_Net1 = config[:141]
    config_Net2 = config[141:]
    #default play
    if len(agent_memory)<1:
        play = 50
    #play as RegNet1
    elif len(agent_memory) ==1:
        X = np.array(agent_memory[-1]).reshape((-1, 1))
        play = forward(config_Net1, X, layers=[1, 10, 10, 1])[0, 0]
    #play as RegNet2
    else:
        X = np.array(agent_memory[-2:]).reshape((-1,1))
        play = forward(config_Net2, X, layers=[2, 10, 10, 1])[0,0]
    return play
