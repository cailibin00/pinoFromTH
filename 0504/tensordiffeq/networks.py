import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations,constraints,initializers,regularizers


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

import numpy as np

# define the baseline FC neural network model
# information about how to define custom neural networks is available
# in the docs - https://docs.tensordiffeq.io/hacks/networks/index.html
def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"))
    model.add(layers.Dense(
        layer_sizes[-1], activation=None,
        kernel_initializer="glorot_normal"))
    return model

def new_neural_net_Exact_D_BC(layer_sizes,bc_val):

    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor

    # A layer instance is callable on a tensor, and returns a tensor.
    #x = layers.InputLayer()(inputs)
    x = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(inputs)
    for width in layer_sizes[2:-1]:
       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x)
    predictions = layers.Dense(1, activation=None, kernel_initializer="glorot_normal")(x)#,bias_initializer=tf.random_normal_initializer(-10,0.001)

    kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1
    kernel_zero_one[1, :] = 1
    inputs_1 = tf.matmul(inputs, kernel_one_zero) #这就是所有的1
    inputs_2 = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2

    sigma_func = (-0.5-inputs_1) * (0.5 - inputs_1) * (-0.5 - inputs_2) * (0.5 - inputs_2) / tf.math.sqrt((-0.5-inputs_1) ** 2 + (0.5 - inputs_1) ** 2 + (-0.5 - inputs_2) ** 2 + (0.5 - inputs_2) ** 2)  # 10 *
    g_func = bc_val#0.072 #0.01/4
    predictions = sigma_func * predictions
    #predictions = layers.Dense(1, activation=None, use_bias=False, kernel_initializer="ones",kernel_constraint=constraints.NonNeg())(predictions)
    predictions = g_func + predictions

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model

def new_neural_net_Exact_D_BC_non_negative(layer_sizes,bc_value):

    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor

    # A layer instance is callable on a tensor, and returns a tensor.
    #x = layers.InputLayer()(inputs)
    x = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(inputs)
    for width in layer_sizes[2:-1]:
       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x)
    predictions = layers.Dense(1, activation=None, kernel_initializer="glorot_normal",bias_initializer="glorot_normal")(x)#tf.random_normal_initializer(10,1)"glorot_normal"

    kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1
    kernel_zero_one[1, :] = 1
    inputs_1 = tf.matmul(inputs, kernel_one_zero) #这就是所有的1
    inputs_2 = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2

    sigma_func = (-0.5 - inputs_1) * (0.5 - inputs_1) * (-0.5 - inputs_2) * (0.5 - inputs_2) / tf.math.sqrt(
        (-0.5 - inputs_1) ** 2 + (0.5 - inputs_1) ** 2 + (-0.5 - inputs_2) ** 2 + (0.5 - inputs_2) ** 2)  # 10 *#########################
    g_func = tf.math.sqrt(bc_value)  #0.01/4 / 4, 1/4)0.072
    #Squareplus_layer = Squareplus(b=1e-12)

    #g_func = (4*(0.072)**2-1/Squareplus_layer.b)/(4*0.072)

    predictions = sigma_func * predictions
    #predictions = layers.Dense(1, activation=None, use_bias=False, kernel_initializer="ones",kernel_constraint=constraints.NonNeg())(predictions)
    predictions = g_func + predictions

    #predictions = ((predictions) ** 2 + tf.math.abs(predictions) * predictions) / 2
    #predictions = tf.nn.relu(predictions)**2
    predictions = predictions*tf.nn.relu(predictions)
    #predictions = Squareplus_layer(predictions)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    return model

def new_neural_net_Exact_D_BC_two_output(layer_sizes,bc_value):#simple example JFO

    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor
    # A layer instance is callable on a tensor, and returns a tensor.
    x = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(inputs)
    for width in layer_sizes[2:-1]:
       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x)
    predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.constant_initializer(0), kernel_initializer="glorot_normal")(x)#(0,-5)tf.random_normal_initializer(0,0.1)
    #use_bias=True,bias_initializer=tf.random_normal_initializer(10,0.1),tf.random_normal_initializer(0,0.1)
    #x = layers.Dense(layer_sizes[-2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    #prediction_g = layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True,bias_initializer=tf.constant_initializer(-7),kernel_initializer="glorot_normal")(x)  # tf.nn.sigmoid-5tf.random_normal_initializer(-5, 0.1)
    #tf.random_normal_initializer(-5, 0.1)tf.random_normal_initializer(-5, 0.01)
    prediction_g = layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True,bias_initializer=tf.constant_initializer(0), kernel_initializer="glorot_normal")(x)#-5
    #-3   tf.constant_initializer(-20)
    #prediction_g = tf.nn.relu6((prediction_g - 1e-5) * 6/(1.-1e-5))/6

    #prediction_g = tf.nn.relu6(prediction_g) / 6

    kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1
    kernel_zero_one[1, :] = 1
    inputs_1 = tf.matmul(inputs, kernel_one_zero) #这就是所有的1
    inputs_2 = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2

    #sigma_func = (inputs_1) * (1 - inputs_1) * (inputs_2) * (1 - inputs_2) / tf.math.sqrt((inputs_1) ** 2 + (1 - inputs_1) ** 2 + (inputs_2) ** 2 + (1 - inputs_2) ** 2)#10 *
    sigma_func = (-0.5-inputs_1) * (0.5 - inputs_1) * (-0.5 - inputs_2) * (0.5 - inputs_2) / tf.math.sqrt(
        (-0.5-inputs_1) ** 2 + (0.5 - inputs_1) ** 2 + (-0.5 - inputs_2) ** 2 + (0.5 - inputs_2) ** 2)  # 10 *
    predictions = sigma_func * predictions
    #predictions = layers.Dense(1, activation=None,use_bias=False, kernel_initializer="ones",kernel_constraint=constraints.NonNeg())(predictions)
    #g_func = tf.math.sqrt(0.072)  # 0.01/4
    g_func = tf.math.sqrt(bc_value)###################################################这里用tf.math的话会让lbfgs变成Nan
    #g_func = bc_value

    #Squareplus_layer = Squareplus(b=1e-12)
    #g_func = 0.072 #0.01/4
    #g_func = (4*(0.072)**2-Squareplus_layer.b)/(4*0.072)
    #g_func = tf.math.log(tf.math.exp(0.072)-1)
    #g_func = -tf.math.log((1/(0.072)-1))  # 0.01/4

    predictions = g_func + predictions
    predictions = tf.nn.relu(predictions)**2
    #predictions = predictions*tf.nn.relu(predictions)
    #predictions = tf.nn.sigmoid(predictions)
    #predictions = tf.nn.softplus(predictions)

    #predictions = Squareplus_layer(predictions)

    predictions_all = [predictions,prediction_g]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model

def new_neural_period_polar_exactBC_old(layer_sizes,bc_values): # test############################
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor
    #inputs = MaxMin_layer()(inputs)

    kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1
    kernel_zero_one[1, :] = 1
    inputs_r = tf.matmul(inputs, kernel_one_zero) #这就是所有的1#r
    inputs_theta = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2#theta



    #inputs_new = Coslayer(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs_new)

    #x = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(inputs_new)

    x = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(inputs)

    for width in layer_sizes[2:-1]:

       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x) #+ x

    #x_1 = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    #x_2 = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    #predictions = layers.Dense(1, activation=None, kernel_initializer="glorot_normal",bias_initializer="glorot_normal")(x)
    predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer=tf.keras.initializers.HeNormal())(x)

    #prediction_g = tf.nn.relu(tf.nn.sigmoid(prediction_g)-1e-5)#1e-6

    sigma_func = (1. - inputs_r)*(1. - inputs_theta) * (1. + inputs_r)*(1. + inputs_theta)/ tf.math.sqrt((1. - inputs_r) ** 2 + (1. + inputs_r) ** 2+(1. - inputs_theta) ** 2 + (1. + inputs_theta) ** 2)  # 10 *
    #(1-inputs_r) * (1+inputs_r)/ tf.math.sqrt((1-inputs_r) ** 2 + (1 + inputs_r) ** 2)  # 10 *


    #g_func = (19.751542067604245-1)/(1/9)*(inputs_r-1)+1#0.072 #0.01/4
    #g_func = (1 - 0.05062895831511623) * (inputs_r + 1/2) + 0.05062895831511623  # 0.072 #0.01/4
    g_func = (bc_values[1] - bc_values[0])/2.0 * (inputs_r + 1.0) + bc_values[0]  # 0.072 #0.01/4

    predictions = g_func + sigma_func * predictions


    predictions_all = [predictions]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model

def new_neural_normalization(layer_sizes,r_lim,theta_lim):
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor
    #inputs = MaxMin_layer()(inputs)

    kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1.0
    kernel_zero_one[1, :] = 1.0
    inputs_r = tf.matmul(inputs, kernel_one_zero) #这就是所有的1#r
    inputs_theta = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2#theta

    inputs_R = 2.0*(inputs_r-r_lim[0])/(r_lim[1]-r_lim[0])-1.0
    inputs_Theta = 2.0 * (inputs_theta - theta_lim[0]) / (theta_lim[1] - theta_lim[0]) - 1.0

    inputs_new = tf.concat([inputs_R, inputs_Theta], 1)

    x = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(inputs_new)
    for width in layer_sizes[3:-1]:

       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x) #+ x


    predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer=tf.keras.initializers.HeNormal())(x)
    predictions_2 = layers.Dense(1, activation=None, use_bias=True, bias_initializer=tf.keras.initializers.HeNormal(),kernel_initializer=tf.keras.initializers.HeNormal())(x)
    predictions_all = [predictions,predictions_2]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model

def new_neural_period_polar_exactBC(layer_sizes,bc_values,r_lim,theta_lim):
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor
    #inputs = MaxMin_layer()(inputs)

    kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1.0
    kernel_zero_one[1, :] = 1.0
    inputs_r = tf.matmul(inputs, kernel_one_zero) #这就是所有的1#r
    inputs_theta = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2#theta



    inputs_R = 2.0*(inputs_r-r_lim[0])/(r_lim[1]-r_lim[0])-1.0
    inputs_Theta = 2.0 * (inputs_theta - theta_lim[0]) / (theta_lim[1] - theta_lim[0]) - 1.0

    inputs_new = tf.concat([inputs_R, inputs_Theta], 1)

    inputs_new = Coslayer(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs_new)

    x = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(inputs_new)
    for width in layer_sizes[3:-1]:

       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x) #+ x

    #predictions = layers.Dense(1, activation=None, kernel_initializer="glorot_normal",bias_initializer="glorot_normal")(x)
    predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer=tf.keras.initializers.HeNormal())(x)

    #prediction_g = tf.nn.relu(tf.nn.sigmoid(prediction_g)-1e-5)#1e-6

    sigma_func = (1. - inputs_R) * (1. + inputs_R)/ tf.math.sqrt((1. - inputs_R) ** 2 + (1. + inputs_R) ** 2)  # 10 *

    g_func = (bc_values[1] - bc_values[0])/2.0 * (inputs_R + 1.0) + bc_values[0]  # 0.072 #0.01/4

    predictions = g_func + sigma_func * predictions


    predictions_all = [predictions]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model

def new_neural_period_polar_exactBC_two_output_one(layer_sizes,bc_values,r_lim,theta_lim):
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor

    '''kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1.0
    kernel_zero_one[1, :] = 1.0

    inputs_r = tf.matmul(inputs, kernel_one_zero) #这就是所有的1#r
    inputs_theta = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2#theta

    inputs_R = 2.0*(inputs_r-r_lim[0])/(r_lim[1]-r_lim[0])-1.0
    inputs_Theta = 2.0 * (inputs_theta - theta_lim[0]) / (theta_lim[1] - theta_lim[0]) - 1.0

    inputs_new = tf.concat([inputs_R, inputs_Theta], 1)'''

    x,inputs_R = Coslayer_normalization(layer_sizes[1],r_lim,theta_lim, activation=tf.nn.tanh, kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs)
    #x = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_U = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_V = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)

    for width in layer_sizes[1:-1]:
        x_t = layers.Dense(width, activation=None, kernel_initializer="glorot_normal")(x)
        x = tf.nn.tanh(x_t*x_U + (1-x_t)*x_V) #+ x
        #x_t = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
        #x = x_t*x_U + (1-x_t)*x_V #+ x
        #x = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)


    '''for width in layer_sizes[1:-1]:
       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x)
       x = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x) + x
    '''
    predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer=tf.keras.initializers.HeNormal())(x)
    #prediction_g = layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True, bias_initializer=tf.constant_initializer(-5), kernel_initializer="glorot_normal")(x)  # -5
    #prediction_g = tf.nn.relu(tf.nn.sigmoid(prediction_g)-1e-5)#1e-6

    sigma_func = 1*(1. - inputs_R) * (1. + inputs_R)/ tf.math.sqrt((1. - inputs_R) ** 2 + (1. + inputs_R) ** 2)  # ADF
    #g_func = tf.math.sqrt((bc_values[1] - bc_values[0])/2.0 * (inputs_R + 1.0) + bc_values[0])  #BC_func
    g_func = ((bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[0])
    predictions = g_func + sigma_func * predictions

    #predictions = predictions * tf.nn.relu(predictions) #

    predictions_all = [predictions,predictions*0]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model

def new_neural_period_polar_exactBC_two_output_three(layer_sizes,bc_values,r_lim,theta_lim):
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor


    x,inputs_R = Coslayer_normalization(layer_sizes[1],r_lim,theta_lim, activation=tf.nn.tanh, kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs)
    #x = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_U = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_V = layers.Dense(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)

    for width in layer_sizes[1:-1]:
        x_t = layers.Dense(width, activation=None, kernel_initializer="glorot_normal")(x)
        x = tf.nn.tanh(x_t*x_U + (1-x_t)*x_V) #+ x
        #x_t = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
        #x = x_t*x_U + (1-x_t)*x_V #+ x
        #x = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)


    '''for width in layer_sizes[1:-1]:
       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x)
       x = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x) + x
    '''
    predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer=tf.keras.initializers.HeNormal())(x)
    predictions_Q_R = layers.Dense(1, activation=None, use_bias=True, bias_initializer=tf.keras.initializers.HeNormal(),kernel_initializer=tf.keras.initializers.HeNormal())(x)
    predictions_Q_Theta = layers.Dense(1, activation=None, use_bias=True, bias_initializer=tf.keras.initializers.HeNormal(),kernel_initializer=tf.keras.initializers.HeNormal())(x)
    #prediction_g = layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True, bias_initializer=tf.constant_initializer(-5), kernel_initializer="glorot_normal")(x)  # -5
    #prediction_g = tf.nn.relu(tf.nn.sigmoid(prediction_g)-1e-5)#1e-6

    sigma_func = 1*(1. - inputs_R) * (1. + inputs_R)/ tf.math.sqrt((1. - inputs_R) ** 2 + (1. + inputs_R) ** 2)  # ADF
    #g_func = tf.math.sqrt((bc_values[1] - bc_values[0])/2.0 * (inputs_R + 1.0) + bc_values[0])  #BC_func
    g_func = ((bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[0])
    predictions = g_func + sigma_func * predictions

    #predictions = predictions * tf.nn.relu(predictions) #

    predictions_all = [predictions,predictions_Q_R,predictions_Q_Theta]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model


def new_neural_period_polar_exactBC(layer_sizes, bc_values, r_lim, theta_lim):
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor

    x, inputs_R = Coslayer_normalization(layer_sizes[1], r_lim, theta_lim, activation=tf.nn.tanh,
                                         kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs)
    # x = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_U = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_V = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)

    for width in layer_sizes[2:-1]:
        x_t = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)  # None
        x = (x_t * x_U + (1 - x_t) * x_V)  # + x
        # x_t = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
        # x = x_t*x_U + (1-x_t)*x_V #+ x
        # x = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)

    predictions = layers.Dense(1, activation=None, use_bias=True, bias_initializer="glorot_normal",
                               kernel_initializer="glorot_normal")(x)


    N = 2
    sigma_func = (1. - inputs_R) * (1. + inputs_R) / ((1. - inputs_R) ** N + (1. + inputs_R) ** N) ** (1 / N)
    # g_func = tf.math.atanh(tf.math.sqrt((bc_values[1] - bc_values[0])/2.0 * (inputs_R + 1.0) + bc_values[0]))  #BC_func  tf.math.atanh  tf.math.sqrt tf.math.atanh
    g_func = (tf.math.pow((bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[0], 1))#tf.math.atanh 1 / 2
    predictions = g_func + sigma_func * predictions

    # prediction_g = -10 + sigma_func * prediction_g
    # prediction_g = tf.nn.sigmoid(prediction_g)

    # prediction_g = tf.nn.leaky_relu(tf.nn.tanh(prediction_g),alpha=1e-1)
    # prediction_g = prediction_g * tf.nn.relu(prediction_g)
    # prediction_g = tf.nn.tanh(prediction_g)**2
    #predictions = (tf.nn.tanh(predictions)) ** 2
    #predictions = predictions * tf.nn.relu(predictions)
    # predictions = tf.nn.relu(predictions)  #
    # predictions = predictions * tf.nn.leaky_relu(predictions,alpha=0.1)

    predictions_all = [predictions,0*predictions]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model


def new_neural_period_polar_exactBC_two_output(layer_sizes, bc_values, r_lim, theta_lim,
                                                activation="tanh", use_residual=False,
                                                output_head_dim=64, coslayer_mode="simple",
                                                gamma_output_transform="tanh_square"):
    """
    Main PINN architecture for Reynolds equation with JFO cavitation.

    Args:
        layer_sizes: [in_dim, cos_units, hidden_0, ..., hidden_N, out_dim]
        bc_values:   [bc_lower, bc_upper]
        r_lim:       [r_min, r_max]
        theta_lim:   [theta_min, theta_max]
        activation:  "tanh" or "silu" — activation for U/V/gate/output branches
        use_residual: whether to use residual skip connections
        output_head_dim: hidden dim inside deep output heads
        coslayer_mode: "simple" (original linear mix) or "mlp" (separate R/θ MLP pathways)
    """
    act_fn = tf.nn.tanh if activation == "tanh" else tf.nn.silu

    inputs = tf.keras.Input(shape=(2,))

    x, inputs_R = Coslayer_normalization(
        layer_sizes[1], r_lim, theta_lim,
        activation=tf.nn.tanh,
        kernel_initializer="glorot_normal",
        bias_initializer=tf.constant_initializer(0),
        coslayer_mode=coslayer_mode
    )(inputs)

    # U / V base branches
    base_width = layer_sizes[2]
    x_U = layers.Dense(base_width, activation=act_fn, kernel_initializer="glorot_normal")(x)
    x_V = layers.Dense(base_width, activation=act_fn, kernel_initializer="glorot_normal")(x)

    # Hermite interpolation branches
    x_1 = layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_2 = layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    #x_3 = layers.Dense(1, activation=None, kernel_initializer="glorot_normal")(x)  # None
    #x_4 = layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)  # None
    #x_5 = layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)  # None

    # Gated hidden layers with optional residual connections
    hidden_w = list(layer_sizes[2:-1])
    prev = None

    for i, w in enumerate(hidden_w):
        # Gate from current x
        x_t = layers.Dense(w, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)

        # U / V at current width (with projection if needed)
        if w == base_width:
            U_w, V_w = x_U, x_V
        else:
            U_w = layers.Dense(w, activation=act_fn, kernel_initializer="glorot_normal")(x_U)
            V_w = layers.Dense(w, activation=act_fn, kernel_initializer="glorot_normal")(x_V)

        main = x_t * U_w + (1.0 - x_t) * V_w

        # Residual skip
        if prev is not None and use_residual:
            if prev.shape[-1] != w:
                skip = layers.Dense(w, activation=None, kernel_initializer="glorot_normal")(prev)
            else:
                skip = prev
            main = main + skip

        prev = main
        x = main

    # x is now the last hidden state [N, last_dim]
    last_dim = hidden_w[-1]
    H = output_head_dim

    # ---- Deep Output Head: Pressure ----
    # last_dim -> H -> (H->H residual) -> 1
    p_h1 = layers.Dense(H, activation=act_fn, kernel_initializer="glorot_normal")(x)
    p_h2 = layers.Dense(H, activation=act_fn, kernel_initializer="glorot_normal")(p_h1)
    if use_residual:
        p_h2 = p_h2 + p_h1
    p_raw = layers.Dense(1, activation=None, use_bias=True,
                         bias_initializer=tf.constant_initializer(0),
                         kernel_initializer="glorot_normal")(p_h2)

    # ---- Deep Output Head: Gamma ----
    # Stage 1: last_dim -> H -> (H->H residual)
    g_h1 = layers.Dense(H, activation=act_fn, kernel_initializer="glorot_normal")(x)
    g_h2 = layers.Dense(H, activation=act_fn, kernel_initializer="glorot_normal")(g_h1)
    if use_residual:
        g_h2 = g_h2 + g_h1

    # Stage 2: concat(pressure_hidden, gamma_hidden) -> H -> (H->H residual) -> 1
    g_cat = tf.concat([p_h2, g_h2], axis=1)
    g_cat1 = layers.Dense(H, activation=act_fn, kernel_initializer="glorot_normal")(g_cat)
    g_cat2 = layers.Dense(H, activation=act_fn, kernel_initializer="glorot_normal")(g_cat1)
    if use_residual:
        g_cat2 = g_cat2 + g_cat1
    g_raw = layers.Dense(1, activation=None, use_bias=True,
                         bias_initializer=tf.constant_initializer(0),
                         kernel_initializer=tf.constant_initializer(1e-6))(g_cat2)

    #N=2
    #sigma_func = (1. - inputs_R) * (1. + inputs_R) / ((1. - inputs_R) ** N + (1. + inputs_R) ** N)**(1/N)
    sigma_func_1 = Out_Imp_BC_layer(para_exp_BC_initializer=tf.constant_initializer(1.))(inputs_R)
    sigma_func_2 = Out_Imp_BC_layer(para_exp_BC_initializer=tf.constant_initializer(1.))(inputs_R)
    #sigma_func_1_withNN = x_3 * (1. - tf.math.exp(x_4 * (-1. - inputs_R))) * (1. - tf.math.exp(x_5 * (inputs_R - 1.)))
    #sigma_func_1 = self.para_exp_BC_3 * (1. - tf.math.exp(self.para_exp_BC_1 * (-1. - inputs_R))) * (1. - tf.math.exp(self.para_exp_BC_2 * (inputs_R - 1.)))

    #g_func = (tf.math.sqrt((bc_values[1] - bc_values[0])/2.0 * (inputs_R + 1.0) + bc_values[0]))  #BC_func  tf.math.atanh  tf.math.sqrt tf.math.atanh
    #g_func = tf.math.atanh(tf.math.pow((bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[0],1/2))

    g_func_1 = tf.math.atanh((tf.math.sqrt(bc_values[1]) - tf.math.sqrt(bc_values[0])) / 2.0 * (inputs_R + 1.0) + tf.math.sqrt(bc_values[0]))
    #g_func = tf.math.atanh((bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[0])
    #g_func = (1-inputs_R)/(2)*bc_values[0] + (inputs_R + 1)/2*bc_values[1]#一次Lagrange插值多项式
        #tf.math.atanh((bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[0])*

    bc_values_transform =np.arctan(np.sqrt(bc_values))
    #g_func_2 = Out_Imp_BC_value_layer(bc_values = bc_values_transform,para_Hermite_BC_initializer=tf.constant_initializer((bc_values_transform[1]-bc_values_transform[0])/2.))(inputs_R)#二次哈密顿插值
    #g_func_2 = Out_Imp_BC_value_layer(bc_values=[0.,0.],para_Hermite_BC_initializer=tf.constant_initializer(0.))(inputs_R)  # 二次哈密顿插值

    g_func_2 = x_1 * (inputs_R + 1) * ((inputs_R - 1) / (-2)) ** 2 + x_2 * (inputs_R - 1) * ((inputs_R + 1) / (2)) ** 2

    g_func = g_func_1 + g_func_2

    # BC enforcement
    predictions = g_func + sigma_func_1 * p_raw

    predictions = tf.nn.tanh(predictions) ** 2
    if gamma_output_transform == "sigmoid":
        prediction_g = sigma_func_2 * tf.nn.sigmoid(g_raw)
    elif gamma_output_transform == "tanh_square":
        prediction_g = tf.nn.tanh(sigma_func_2 * g_raw) ** 2
    else:
        raise ValueError(
            "gamma_output_transform must be 'tanh_square' or 'sigmoid', "
            f"got {gamma_output_transform!r}"
        )

    predictions_all = [predictions, prediction_g]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model


def new_neural_period_polar_exactBC_two_output_is_texture_func(layer_sizes, bc_values, r_lim, theta_lim):
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor


    x, inputs_R = Coslayer_normalization(layer_sizes[1], r_lim, theta_lim, activation=tf.nn.tanh,
                                         kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs)
    # x = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_U = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_V = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)

    for width in layer_sizes[2:-1]:
        x_t = layers.Dense(width, activation=None, kernel_initializer="glorot_normal")(x)
        x = tf.nn.tanh(x_t * x_U + (1 - x_t) * x_V)  # + x
        # x_t = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
        # x = x_t*x_U + (1-x_t)*x_V #+ x
        # x = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    '''

    x_t_1 = layers.Dense(width, activation=None, kernel_initializer="glorot_normal")(x)
    x_1 = tf.nn.tanh(x_t_1*x_U + (1-x_t_1)*x_V) #+ x

    x_t_2 = layers.Dense(width, activation=None, kernel_initializer="glorot_normal")(x)
    x_2 = tf.nn.tanh(x_t_2*x_U + (1-x_t_2)*x_V) #+ x
    '''

    '''for width in layer_sizes[1:-1]:
       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x)
       x = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x) + x
    '''
    predictions = layers.Dense(1, activation=None, use_bias=True, bias_initializer="glorot_normal",
                               kernel_initializer="glorot_normal")(x)

    prediction_g = layers.Dense(1, activation=None, use_bias=True, bias_initializer="glorot_normal",
                                kernel_initializer="glorot_normal")(x)  # -5

    # predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer=tf.keras.initializers.HeNormal())(x)
    # prediction_g = layers.Dense(1, activation=None, use_bias=True, bias_initializer=tf.keras.initializers.HeNormal(),kernel_initializer=tf.keras.initializers.HeNormal())(x)  # -5

    #
    sigma_func = 1 * (1. - inputs_R) * (1. + inputs_R) / tf.math.sqrt(
        (1. - inputs_R) ** 2 + (1. + inputs_R) ** 2)  # ADF
    g_func = tf.math.atanh(tf.math.sqrt((bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[
        0]))  # BC_func  tf.math.atanh  tf.math.sqrt tf.math.atanh
    predictions = g_func + sigma_func * predictions

    # prediction_g = -10 + sigma_func * prediction_g
    prediction_g = tf.nn.sigmoid(prediction_g)
    #prediction_g =tf.nn.tanh(prediction_g)**2
    #prediction_g = sigma_func * prediction_g
    #prediction_g = (tf.nn.tanh(prediction_g)) ** 2

    # prediction_g = tf.nn.leaky_relu(tf.nn.tanh(prediction_g),alpha=1e-1)
    # prediction_g = prediction_g * tf.nn.relu(prediction_g)
    # prediction_g = tf.nn.tanh(prediction_g)**2
    predictions = tf.nn.tanh(predictions)  # **2
    predictions = predictions * tf.nn.relu(predictions)
    # predictions = tf.nn.relu(predictions)  #
    # predictions = predictions * tf.nn.leaky_relu(predictions,alpha=0.01)

    predictions_all = [predictions, prediction_g]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model

def new_neural_period_polar_exactBC_two_output_new(layer_sizes,bc_values,r_lim,theta_lim):

        inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor
        inputs_Theta, inputs_R = Coslayer_normalization_new(layer_sizes[1], r_lim, theta_lim, activation=tf.nn.tanh,
                                                 kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs)
        inputs_new = [inputs_Theta, inputs_R]
        outputs,xy,pred = [],[],[]


        for X in inputs_new:
                U = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(X)
                V = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(X)
                H = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(X)

                for width in layer_sizes[2:-2]:
                    Z = layers.Dense(width, activation=None, kernel_initializer="glorot_normal")(H)
                    H = tf.nn.tanh(Z * U + (1 - Z) * V)  # + x outputs += [tf.transpose(H, (1, 0))]

                H_1 = layers.Dense(layer_sizes[-2], activation=None, kernel_initializer="glorot_normal")(H)
                H_2 = layers.Dense(layer_sizes[-2], activation=None, kernel_initializer="glorot_normal")(H)

                outputs += [H_1,H_2]
                #outputs += [tf.transpose(H, (1, 0))]
        #pred = [tf.reduce_sum(outputs[0]*outputs[2]),tf.reduce_sum(outputs[1]*outputs[3])]
        pred = [tf.matmul(outputs[0] , tf.transpose(outputs[2])), tf.matmul(outputs[1] , tf.transpose(outputs[3]))]
        #for i in range(layer_sizes[-1]):
        #    pred += [tf.matmul( outputs[0][layer_sizes[-2] * i:layer_sizes[-2] * (i + 1)],tf.transpose(outputs[1][layer_sizes[-2] * i:layer_sizes[-2] * (i + 1)]))]
            #pred += [tf.einsum('fx, fy->fxy', outputs[0][layer_sizes[-2]*i:layer_sizes[-2]*(i+1)], outputs[1][layer_sizes[-2]*i:layer_sizes[-2]*(i+1)])]
            #pred += [tf.einsum('fxy, fz->xyz', xy[i], outputs[-1][layer_sizes[-2]*i:layer_sizes[-2]*(i+1)])]

        sigma_func = 1 * (1. - inputs_R) * (1. + inputs_R) / tf.math.sqrt((1. - inputs_R) ** 2 + (1. + inputs_R) ** 2)  # ADF
        g_func = tf.math.atanh(tf.math.sqrt((bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[0]))  # BC_func  tf.math.atanh  tf.math.sqrt
        predictions = pred[0]
        prediction_g = pred[1]

        predictions = g_func + sigma_func * predictions

        prediction_g = sigma_func * prediction_g
        prediction_g = tf.nn.tanh(prediction_g) ** 2

        predictions = tf.nn.tanh(predictions)  # **2
        predictions = predictions * tf.nn.relu(predictions)  #

        predictions_all = [predictions, prediction_g]

        model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
        return model

def new_neural_period_polar_exactBC_two_output_old(layer_sizes,bc_values,r_lim,theta_lim):
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor

    kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1.0
    kernel_zero_one[1, :] = 1.0

    inputs_r = tf.matmul(inputs, kernel_one_zero) #这就是所有的1#r
    inputs_theta = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2#theta

    inputs_R = 2.0*(inputs_r-r_lim[0])/(r_lim[1]-r_lim[0])-1.0
    inputs_Theta = 2.0 * (inputs_theta - theta_lim[0]) / (theta_lim[1] - theta_lim[0]) - 1.0

    inputs_new = tf.concat([inputs_R, inputs_Theta], 1)

    x = Coslayer(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs_new)
    for width in layer_sizes[2:-1]:
       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x) + x

    predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer=tf.keras.initializers.HeNormal())(x)
    prediction_g = layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True, bias_initializer=tf.constant_initializer(0), kernel_initializer="glorot_normal")(x)  # -5
    #prediction_g = tf.nn.relu(tf.nn.sigmoid(prediction_g)-1e-5)#1e-6

    sigma_func = (1. - inputs_R) * (1. + inputs_R)/ tf.math.sqrt((1. - inputs_R) ** 2 + (1. + inputs_R) ** 2)  # 10 *

    g_func = tf.math.sqrt((bc_values[1] - bc_values[0])/2.0 * (inputs_R + 1.0) + bc_values[0])  # 0.072 #0.01/4
    #g_func = (tf.math.sqrt(bc_values[1]) - tf.math.sqrt(bc_values[0]) / 2.0 * (inputs_R + 1.0) + tf.math.sqrt(bc_values[0]))
    #g_func = (bc_values[1] - bc_values[0]) / 2.0 * (inputs_R + 1.0) + bc_values[0]

    predictions = g_func + sigma_func * predictions
    predictions = predictions * tf.nn.relu(predictions) #

    predictions_all = [predictions,prediction_g]
    model = tf.keras.Model(inputs=inputs, outputs=predictions_all)
    return model

def new_neural_H(layer_sizes,bc_values,r_lim,theta_lim):
    inputs = tf.keras.Input(shape=(2,))  # Returns a placeholder tensor

    kernel_one_zero = np.zeros((2, 1))
    kernel_zero_one = np.zeros((2, 1))
    kernel_one_zero[0, :] = 1.0
    kernel_zero_one[1, :] = 1.0

    inputs_r = tf.matmul(inputs, kernel_one_zero) #这就是所有的1#r
    inputs_theta = tf.matmul(inputs, kernel_zero_one)  # 这就是所有的2#theta

    inputs_R = 2.0*(inputs_r-r_lim[0])/(r_lim[1]-r_lim[0])-1.0
    inputs_Theta = 2.0 * (inputs_theta - theta_lim[0]) / (theta_lim[1] - theta_lim[0]) - 1.0

    inputs_new = tf.concat([inputs_R, inputs_Theta], 1)

    x = Coslayer(layer_sizes[1], activation=tf.nn.tanh, kernel_initializer="glorot_normal", bias_initializer="glorot_normal")(inputs_new)
    for width in layer_sizes[2:-1]:
       x = layers.Dense(width, activation=tf.nn.tanh,kernel_initializer="glorot_normal")(x) + x
    output = x
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

class Coslayer_normalization(layers.Layer):
    """
    Fourier feature encoding layer with two modes:

    - "simple" (original TF):  tanh(kernel_R * R_norm + kernel_theta * cos(pi*theta_norm + phi) + bias)
    - "mlp" (PyTorch-style):   Separate R / theta MLP pathways before fusion:
         theta:  cos(pi*theta_norm + phi)  ->  FC(units)->SiLU -> FC(half)->SiLU -> theta_feat
         R:      R_norm                     ->  FC(units)->SiLU -> FC(half)->SiLU -> R_feat
         output: concat(theta_feat, R_feat) -> activation
    """

    def __init__(self,
                 units,
                 r_lim,
                 theta_lim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 coslayer_mode="simple",
                 **kwargs):
        super(Coslayer_normalization, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(f'Received an invalid value for `units`, expected '
                             f'a positive integer, got {units}.')

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.r_lim = r_lim
        self.theta_lim = theta_lim
        self.coslayer_mode = coslayer_mode

        self.kernel_one_zero = np.zeros((2, 1))
        self.kernel_zero_one = np.zeros((2, 1))
        self.kernel_one_zero[0, :] = 1.0
        self.kernel_zero_one[1, :] = 1.0

        self.supports_masking = True

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))

        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        if self.coslayer_mode == "mlp":
            # ---- MLP mode: separate R/theta pathways ----
            half = self.units // 2

            # Fourier phase (shared)
            self.phy = self.add_weight(
                'phy_cos',
                shape=[self.units, ],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

            self.K_const = tf.constant(np.pi, shape=[1, ], dtype=self.dtype)

            # theta pathway:  units -> units -> half
            self.theta_fc1 = layers.Dense(self.units, activation=tf.nn.silu,
                                          kernel_initializer="glorot_normal",
                                          name='theta_fc1')
            self.theta_fc2 = layers.Dense(half, activation=tf.nn.silu,
                                          kernel_initializer="glorot_normal",
                                          name='theta_fc2')

            # R pathway:  1 -> units -> half
            self.r_fc1 = layers.Dense(self.units, activation=tf.nn.silu,
                                      kernel_initializer="glorot_normal",
                                      name='r_fc1')
            self.r_fc2 = layers.Dense(half, activation=tf.nn.silu,
                                      kernel_initializer="glorot_normal",
                                      name='r_fc2')
        else:
            # ---- Simple mode: original linear combination ----
            self.kernel = self.add_weight(
                'kernel_cos',
                shape=[last_dim, self.units],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

            self.K_const = tf.constant(np.pi, shape=[1, ], dtype=self.dtype)

            self.phy = self.add_weight(
                'phy_cos',
                shape=[self.units, ],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)

            if self.use_bias:
                self.bias = self.add_weight(
                    'bias_cos',
                    shape=[self.units, ],
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    dtype=self.dtype,
                    trainable=True)
            else:
                self.bias = None

        self.built = True

    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        inputs_r = tf.matmul(inputs, self.kernel_one_zero)   # [N, 1]
        inputs_theta = tf.matmul(inputs, self.kernel_zero_one)  # [N, 1]

        inputs_R = 2.0 * (inputs_r - self.r_lim[0]) / (self.r_lim[1] - self.r_lim[0]) - 1.0
        inputs_Theta = 2.0 * (inputs_theta - self.theta_lim[0]) / (self.theta_lim[1] - self.theta_lim[0]) - 1.0

        if self.coslayer_mode == "mlp":
            # ---- MLP mode ----
            # theta pathway: Fourier -> MLP
            theta_fourier = tf.cos(inputs_Theta * self.K_const + self.phy)  # [N, units]
            theta_feat = self.theta_fc1(theta_fourier)                       # [N, units]
            theta_feat = self.theta_fc2(theta_feat)                          # [N, half]

            # R pathway: raw coordinate -> MLP
            r_feat = self.r_fc1(inputs_R)                                     # [N, units]
            r_feat = self.r_fc2(r_feat)                                       # [N, half]

            # Merge
            outputs = tf.concat([theta_feat, r_feat], axis=1)                 # [N, units]

            if self.activation is not None:
                outputs = self.activation(outputs)
        else:
            # ---- Simple mode: original linear combination ----
            outputs = inputs_Theta * self.K_const                        # [N, 1]
            outputs = tf.add(outputs, self.phy)                          # [N, units]
            outputs = tf.cos(outputs)                                     # [N, units]

            kernel_1 = self.kernel[0, :]   # [units]
            kernel_2 = self.kernel[1, :]   # [units]

            outputs_2 = tf.multiply(outputs, kernel_2)                   # [N, units]
            inputs_r_broadcast = tf.add(inputs_R, 0 * self.phy)          # [N, units]
            outputs_1 = tf.multiply(inputs_r_broadcast, kernel_1)        # [N, units]
            outputs = tf.add(outputs_1, outputs_2)

            if self.use_bias:
                outputs = tf.add(outputs, self.bias)

            if self.activation is not None:
                outputs = self.activation(outputs)

        return outputs, inputs_R

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % (input_shape,))
        return input_shape[:-1].concatenate(self.units)

class Out_Imp_BC_layer(layers.Layer):

        def __init__(self,para_exp_BC_initializer='glorot_uniform', my_param=None, **kwargs):
            self.my_param = my_param
            super(Out_Imp_BC_layer, self).__init__(**kwargs)

            self.para_exp_BC_initializer = initializers.get(para_exp_BC_initializer)

        def build(self, input_shape):

            self.para_exp_BC_1 = self.add_weight(name='para_exp_BC_1',
                                              shape=(1,),
                                              initializer=self.para_exp_BC_initializer,
                                              trainable=True)

            self.para_exp_BC_2 = self.add_weight(name='para_exp_BC_2',
                                              shape=(1,),
                                              initializer=self.para_exp_BC_initializer,
                                              trainable=True)
            self.para_exp_BC_3 = self.add_weight(name='para_exp_BC_3',
                                              shape=(1,),
                                              initializer=self.para_exp_BC_initializer,
                                              trainable=True)

        def call(self, inputs_R):
            #N = 2
            #sigma_func = (1. - inputs_R) * (1. + inputs_R) / ((1. - inputs_R) ** N + (1. + inputs_R) ** N) ** (1 / N)

            sigma_func = self.para_exp_BC_3*(1.-tf.math.exp(self.para_exp_BC_1*(-1. - inputs_R))) * (1.-tf.math.exp(self.para_exp_BC_2*(inputs_R - 1.)))

            return sigma_func

        def compute_output_shape(self, input_shape):
            return input_shape


class Out_Imp_BC_value_layer(layers.Layer):

    def __init__(self,bc_values,para_Hermite_BC_initializer='glorot_uniform', my_param=None, **kwargs):
        self.my_param = my_param
        super(Out_Imp_BC_value_layer, self).__init__(**kwargs)

        self.para_Hermite_BC_initializer = initializers.get(para_Hermite_BC_initializer)
        self.bc_values = bc_values
    def build(self, input_shape):
        self.para_Hermite_BC_1 = self.add_weight(name='para_Hermite_BC_1',
                                             shape=(1,),
                                             initializer=self.para_Hermite_BC_initializer,
                                             trainable=True)

        self.para_Hermite_BC_2 = self.add_weight(name='para_Hermite_BC_2',
                                             shape=(1,),
                                             initializer=self.para_Hermite_BC_initializer,
                                             trainable=True)

    def call(self, inputs_R):
        # N = 2
        # sigma_func = (1. - inputs_R) * (1. + inputs_R) / ((1. - inputs_R) ** N + (1. + inputs_R) ** N) ** (1 / N)
        g_func = self.bc_values[0]*(1 + 2*(inputs_R+1)/  2) *((inputs_R-1)/(-2))**2 \
               + self.bc_values[1]*(1 + 2*(inputs_R-1)/(-2))*((inputs_R+1)/  2)**2  \
               + self.para_Hermite_BC_1*(inputs_R+1)*((inputs_R-1)/(-2))**2    \
               + self.para_Hermite_BC_2*(inputs_R-1)*((inputs_R+1)/( 2))**2    #二点三次Hermite插值多项式

        return g_func

    def compute_output_shape(self, input_shape):
        return input_shape
