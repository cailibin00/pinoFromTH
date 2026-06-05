import numpy as np
import tensorflow as tf


def grad_seperate_all_with_adapt_weight(self):  # 对神经网络系数进行求导
    with tf.GradientTape(persistent=True) as tape:
        loss_all = self.update_loss_seperate()  # 每项loss分别取出来
        # loss_total = self.update_loss()
    if self.PCGrad_true:

        grad_all = [tape.gradient(loss, self.variables) for loss in loss_all]
        np.random.shuffle(grad_all)

        num_tasks = len(loss_all)

        for counter_grad_all in range(num_tasks):
            for counter_grad, grad in enumerate(grad_all[counter_grad_all]):
                if grad == None:
                    grad_all[counter_grad_all][counter_grad] = 0. * grad_total[counter_grad]
        grads_task = [tf.concat([tf.reshape(grad, [-1, ]) for grad in grads], axis=0) for grads in
                      grad_all]  # grad_all_flatten

        grads_task_proj = []
        for grad_task in (grads_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task * grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(
                    grads_task[k] * grads_task[k] + 1e-12)  # 防止当一个任务梯度为0时，出现NaN
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
            grads_task_proj.append(grad_task)
        self.grads_task_proj = grads_task_proj
        proj_grads_flatten = grads_task_proj

        # 重新组合成各层的形式并求和
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(self.variables):
                grad_shape = var.shape
                flatten_dim = int(np.prod(grad_shape))
                proj_grad = proj_grads_flatten[j][start_idx:start_idx + flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(self.variables):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad  # 直接求和了
                start_idx += flatten_dim
        grad_total = proj_grads
        loss_total = tf.reduce_sum(loss_all)
    return loss_total, grad_total, loss_all
