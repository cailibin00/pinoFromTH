import tensorflow as tf
from tensordiffeq.sampling import LHS
import time as time
import numpy as np


def set_weights(model, w, sizes_w, sizes_b):
    """将一维权重向量 w 恢复到 model 的各层。
    跳过没有可训练权重的层（如 Lambda、Normalization 等）。"""
    idx = 0  # 当前在 sizes_w / sizes_b 中的层索引
    for layer in model.layers[0:]:
        layer_weights = layer.get_weights()
        if len(layer_weights) == 0:
            continue  # 跳过无权重层，保持与 get_weights 一致
        if idx >= len(sizes_w):
            break
        start_weights = sum(sizes_w[:idx]) + sum(sizes_b[:idx])
        end_weights = sum(sizes_w[:idx + 1]) + sum(sizes_b[:idx])
        weights = w[start_weights:end_weights]
        w_div = int(sizes_w[idx] / sizes_b[idx])
        weights = tf.reshape(weights, [w_div, sizes_b[idx]])
        biases = w[end_weights:end_weights + sizes_b[idx]]
        weights_biases = [weights, biases]
        layer.set_weights(weights_biases)
        idx += 1


def get_weights(model):
    """获取 model 所有可训练层的权重，展平为一维 tensor。
    跳过没有可训练权重的层。"""
    w = []
    for layer in model.layers[0:]:
        weights_biases = layer.get_weights()
        if len(weights_biases) == 0:
            continue  # 跳过无权重层 (如 coslayer normalization)
        weights = weights_biases[0].flatten()
        biases = weights_biases[1]
        w.extend(weights)
        w.extend(biases)

    w = tf.convert_to_tensor(w)
    return w


def get_sizes_from_model(model, layer_sizes):
    """根据 model 的实际层结构计算 sizes_w 和 sizes_b。
    仅包括有权重层，跳过无权重层（与 get_weights/set_weights 一致）。
    layer_sizes: 主 MLP 的网络结构, 作为后备。

    返回 (sizes_w, sizes_b)，每项为有权重层的 kernel/bias 形状的乘积列表。
    """
    sizes_w = []
    sizes_b = []
    for layer in model.layers[0:]:
        layer_weights = layer.get_weights()
        if len(layer_weights) == 0:
            continue
        w_shape = layer_weights[0].shape  # kernel shape
        b_shape = layer_weights[1].shape  # bias shape
        sizes_w.append(int(np.prod(w_shape)))
        sizes_b.append(int(np.prod(b_shape)))
    return sizes_w, sizes_b


def get_sizes(layer_sizes):
    sizes_w = [layer_sizes[i] * layer_sizes[i - 1] for i in range(len(layer_sizes)) if i != 0]
    sizes_b = layer_sizes[1:]
    return sizes_w, sizes_b


def MSE(pred, actual, weights=None):
    if weights is not None:
        return tf.reduce_mean(tf.square(weights * tf.math.subtract(pred, actual)))
    return tf.reduce_mean(tf.square(tf.math.subtract(pred, actual)))


def g_MSE(pred, actual, g_lam):
    return tf.reduce_mean(g_lam * tf.square(tf.math.subtract(pred, actual)))


def constant(val, dtype=tf.float32):
    return tf.constant(val, dtype=dtype)


def convertTensor(val, dtype=tf.float32):
    return tf.cast(val, dtype=dtype)


def LatinHypercubeSample(N_f, bounds):
    sampling = LHS(xlimits=bounds)
    return sampling(N_f)


def get_tf_model(model):
    return tf.function(model)


def tensor(x, dtype=tf.float32):
    return tf.convert_to_tensor(x, dtype=dtype)


def multimesh(arrs):
    lens = list(map(len, arrs))
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz *= s

    ans = []
    for i, arr in enumerate(arrs):
        slc = [1] * dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)

    return ans  # returns like np.meshgrid


# if desired, this flattens and hstacks the output dimensions for feeding into a tf/keras type neural network
def flatten_and_stack(mesh):
    dims = np.shape(mesh)
    output = np.zeros((len(mesh), np.prod(dims[1:])))
    for i, arr in enumerate(mesh):
        output[i] = arr.flatten()
    return output.T  # returns in an [nxm] matrix


def initialize_weights_loss(init_weights):
    lambdas = []
    lambdas_map = {}
    counter = 0

    for i, (key, values) in enumerate(init_weights.items()):
        list = []
        for value in values:
            if value is not None:
                lambdas.append(tf.Variable(value, trainable=True, dtype=tf.float32))
                list.append(counter)
                counter += 1
        lambdas_map[key.lower()] = list
    return lambdas, lambdas_map
