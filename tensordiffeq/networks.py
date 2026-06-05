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


def new_neural_period_polar_exactBC_two_output(layer_sizes,bc_values,r_lim,theta_lim):
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

    x,inputs_R = Coslayer_normalization(layer_sizes[1],r_lim,theta_lim, activation=tf.nn.tanh, kernel_initializer="glorot_normal", bias_initializer=tf.constant_initializer(0))(inputs)
    #x = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_U = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
    x_V = layers.Dense(layer_sizes[2], activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)


    x_1 = layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)  # None
    x_2 = layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)  # None
    #x_3 = layers.Dense(1, activation=None, kernel_initializer="glorot_normal")(x)  # None
    #x_4 = layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)  # None
    #x_5 = layers.Dense(1, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)  # None

    for width in layer_sizes[2:-1]:
        x_t = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)#None
        x = (x_t*x_U + (1-x_t)*x_V) #+ x

        #x_t = layers.Dense(width, activation=None, kernel_initializer="glorot_normal")(x)#None
        #x = tf.nn.tanh((x_t*x_U + (1-x_t)*x_V)) #+ x

        #x = layers.Dense(width, activation=tf.nn.tanh, kernel_initializer="glorot_normal")(x)
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
    predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.constant_initializer(0), kernel_initializer="glorot_normal")(x)

    prediction_g = layers.Dense(1, activation=None, use_bias=True, bias_initializer=tf.constant_initializer(0),kernel_initializer=tf.constant_initializer(1e-6))(x)  # -5 "glorot_normal" "glorot_normal"tf.constant_initializer(0)

    #predictions = layers.Dense(1, activation=None,use_bias=True,bias_initializer=tf.keras.initializers.HeNormal(), kernel_initializer=tf.keras.initializers.HeNormal())(x)
    #prediction_g = layers.Dense(1, activation=None, use_bias=True, bias_initializer=tf.keras.initializers.HeNormal(),kernel_initializer=tf.keras.initializers.HeNormal())(x)  # -5

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

    g_func_2 = x_1 * (inputs_R + 1) * ((inputs_R - 1) / (-2)) ** 2 + x_2 * (inputs_R - 1) * ((inputs_R + 1) / (2)) ** 2  # 二点三次Hermite插值多项式

    g_func = g_func_1 + g_func_2

    predictions = g_func + sigma_func_1 * predictions
    #predictions = g_func + (sigma_func_1_withNN+sigma_func_1) * predictions
    #prediction_g = -10 + sigma_func * prediction_g

    prediction_g = sigma_func_2 * prediction_g
    #prediction_g = tf.nn.sigmoid(10*prediction_g-20)
    #prediction_g = ((prediction_g)) ** 2
    #prediction_g = (tf.nn.tanh(prediction_g))**2#tf.nn.tanh
    prediction_g = (tf.nn.tanh(prediction_g))**2

    #prediction_g = tf.nn.leaky_relu(tf.nn.tanh(prediction_g),alpha=1e-1)
    #prediction_g = prediction_g * tf.nn.relu(prediction_g)
    #prediction_g = tf.nn.tanh(prediction_g)
    #predictions = tf.nn.tanh(predictions)
    predictions = (tf.nn.tanh(predictions))**2
    #predictions = predictions * tf.nn.relu(predictions)
    #predictions = (predictions)**2  #
    #predictions = predictions * tf.nn.leaky_relu(predictions,alpha=0.1)


    predictions_all = [predictions,prediction_g]
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
        #self.K_initializer = initializers.get(K_initializer)

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

        self.kernel = self.add_weight(
            'kernel_cos',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)

        self.K = tf.constant(np.pi,
            shape=[1, ],
            dtype=self.dtype)#2*

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

        '''kernel_one_zero = np.zeros((2, 1))
        kernel_zero_one = np.zeros((2, 1))
        kernel_one_zero[0, :] = 1.0
        kernel_zero_one[1, :] = 1.0'''

        inputs_r = tf.matmul(inputs, self.kernel_one_zero)  # 这就是所有的1#r [n,1]
        inputs_theta = tf.matmul(inputs, self.kernel_zero_one)  # 这就是所有的2#theta [n,1]
        ########temp############

        inputs_R = 2.0 * (inputs_r - self.r_lim[0]) / (self.r_lim[1] - self.r_lim[0]) - 1.0
        inputs_Theta = 2.0 * (inputs_theta - self.theta_lim[0]) / (self.theta_lim[1] - self.theta_lim[0]) - 1.0

        outputs = inputs_Theta * self.K  # gen_math_ops.MatMul, [n,1]*[1]
        outputs = tf.add(outputs, self.phy)  # nn_ops.bias_add(outputs, self.phy) [n,1]+[m]=[n,m]
        outputs = tf.cos(outputs) #[n,m]
        ########temp############

        #outputs = tf.stack([inputs_r,outputs],axis=1)
        #outputs = outputs[:,:,0]
        #self.kernel [2,m]
        kernel_1 = self.kernel[0, :]#[1,m]
        kernel_2 = self.kernel[1, :]

        outputs_2 = tf.multiply(outputs, kernel_2) #[n,m] [1,m] =  [n,m]# gen_math_ops.MatMul(a=outputs, b=self.kernel)
        inputs_r = tf.add(inputs_R, 0*self.phy) #[n,m]
        outputs_1 = tf.multiply(inputs_R, kernel_1) #[n,1] [1,m] =  [n,m]
        #正常情况下：[n,2]*[2,m]=[n,m]
        outputs = tf.add(outputs_1, outputs_2)
        if self.use_bias:
            outputs = tf.add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs,inputs_R

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % (input_shape,))
        return input_shape[:-1].concatenate(self.units)

# =============================================================================
# Fourier Feature Decoupled Architecture (u_model_switch=13)
# =============================================================================
def new_neural_fourier_decoupled(layer_sizes, bc_values, r_lim, theta_lim,
                                  bc_switch=1, num_freq=4, embed_dim=64):
    """
    Decoupled Fourier feature encoding with separate MLPs for R and θ.
    Keras 3 compatible: all TensorFlow ops on symbolic tensors are wrapped
    inside Keras layers/Lambda.
    """
    inputs = tf.keras.Input(shape=(2,))

    # ── Coordinate extraction ──────────────────────────────────────────
    inputs_r = layers.Lambda(lambda x: x[:, 0:1], name="extract_r")(inputs)
    inputs_theta = layers.Lambda(lambda x: x[:, 1:2], name="extract_theta")(inputs)

    # ── Normalization to [-1, 1] ───────────────────────────────────────
    R_norm = layers.Lambda(
        lambda x: 2.0 * (x - r_lim[0]) / (r_lim[1] - r_lim[0]) - 1.0,
        name="normalize_r"
    )(inputs_r)
    theta_norm = layers.Lambda(
        lambda x: 2.0 * (x - theta_lim[0]) / (theta_lim[1] - theta_lim[0]) - 1.0,
        name="normalize_theta"
    )(inputs_theta)

    # ── Fourier Feature Encoding ───────────────────────────────────────
    R_ff_list, theta_ff_list = [], []
    for i in range(num_freq):
        freq = float((2 ** i) * np.pi)
        R_ff_list.append(layers.Lambda(lambda x, f=freq: tf.sin(f * x))(R_norm))
        R_ff_list.append(layers.Lambda(lambda x, f=freq: tf.cos(f * x))(R_norm))
        theta_ff_list.append(layers.Lambda(lambda x, f=freq: tf.sin(f * x))(theta_norm))
        theta_ff_list.append(layers.Lambda(lambda x, f=freq: tf.cos(f * x))(theta_norm))

    R_ff = layers.Concatenate(axis=1, name="fourier_r")(R_ff_list)
    theta_ff = layers.Concatenate(axis=1, name="fourier_theta")(theta_ff_list)

    # ── R / θ encoders ────────────────────────────────────────────────
    r_h = layers.Dense(32, activation='tanh', kernel_initializer="glorot_normal")(R_ff)
    R_embed = layers.Dense(embed_dim, activation='tanh', kernel_initializer="glorot_normal")(r_h)

    t_h = layers.Dense(32, activation='tanh', kernel_initializer="glorot_normal")(theta_ff)
    theta_embed = layers.Dense(embed_dim, activation='tanh', kernel_initializer="glorot_normal")(t_h)

    x = layers.Concatenate(axis=1, name="merge_embed")([R_embed, theta_embed])

    # ── Main Network with U/V branching ───────────────────────────────
    base_width = layer_sizes[2]
    x_U = layers.Dense(base_width, activation='tanh', kernel_initializer="glorot_normal")(x)
    x_V = layers.Dense(base_width, activation='tanh', kernel_initializer="glorot_normal")(x)
    x = layers.Dense(base_width, activation='tanh', kernel_initializer="glorot_normal")(x)

    for width in layer_sizes[2:-1]:
        if x.shape[-1] != width:
            x = layers.Dense(width, activation='tanh', kernel_initializer="glorot_normal")(x)
            x_U_cur = layers.Dense(width, activation='tanh', kernel_initializer="glorot_normal")(x_U)
            x_V_cur = layers.Dense(width, activation='tanh', kernel_initializer="glorot_normal")(x_V)
        else:
            x_U_cur = x_U
            x_V_cur = x_V

        gate = layers.Dense(width, activation='sigmoid', kernel_initializer="glorot_normal")(x)
        left = layers.Multiply()([gate, x_U_cur])
        right_gate = layers.Lambda(lambda z: 1.0 - z)(gate)
        right = layers.Multiply()([right_gate, x_V_cur])
        x = layers.Add()([left, right])

    # ── Output heads ──────────────────────────────────────────────────
    nn_P = layers.Dense(1, activation=None, use_bias=True,
                        bias_initializer=tf.constant_initializer(0),
                        kernel_initializer="glorot_normal")(x)
    nn_gamma = layers.Dense(1, activation=None, use_bias=True,
                            bias_initializer=tf.constant_initializer(0),
                            kernel_initializer="glorot_normal")(x)

    # ── Boundary condition handling ───────────────────────────────────
    if bc_switch == 1:
        g_h = layers.Dense(8, activation='tanh', kernel_initializer="glorot_normal")(R_norm)
        g_h = layers.Dense(8, activation='tanh', kernel_initializer="glorot_normal")(g_h)
        g_func = layers.Dense(1, activation=None, kernel_initializer="glorot_normal")(g_h)

        transition = 0.03
        t_left = layers.Lambda(
            lambda x: tf.clip_by_value((x + 1.0) / transition, 0.0, 1.0),
            name="clip_left"
        )(R_norm)
        t_right = layers.Lambda(
            lambda x: tf.clip_by_value((1.0 - x) / transition, 0.0, 1.0),
            name="clip_right"
        )(R_norm)
        sigma_func = layers.Lambda(
            lambda xs: (3.0 * xs[0] ** 2 - 2.0 * xs[0] ** 3) *
                       (3.0 * xs[1] ** 2 - 2.0 * xs[1] ** 3),
            name="sigma_func"
        )([t_left, t_right])

        sigma_nn = layers.Multiply()([sigma_func, nn_P])
        P_raw = layers.Add()([g_func, sigma_nn])
        gamma_raw = nn_gamma

    elif bc_switch == 2:
        P_raw = nn_P
        gamma_raw = nn_gamma
    else:
        raise ValueError(f"bc_switch must be 1 or 2, got {bc_switch}")

    P = layers.Lambda(lambda x: tf.square(tf.tanh(x)), name="P_output")(P_raw)
    gamma = layers.Lambda(lambda x: tf.square(tf.tanh(x)), name="gamma_output")(gamma_raw)

    model = tf.keras.Model(inputs=inputs, outputs=[P, gamma])
    return model


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


