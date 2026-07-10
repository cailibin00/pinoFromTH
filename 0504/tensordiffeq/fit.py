import tensorflow as tf
import numpy as np
from .networks import *
from .models import *
from .utils import *
from .optimizers import *
from .output import print_screen
import time
import os
from tqdm.auto import tqdm, trange
from random import random, randint
import sys
from .utils import LatinHypercubeSample

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"


def fit(obj, tf_iter=0, newton_iter=0, newton_eager=False):

    start_time = time.time()

    # these cant be tf.functions on initialization since the distributed strategy requires its own
    # graph using grad and adaptgrad, so they cant be compiled as tf.functions until we know dist/non-dist
    #obj.grad = tf.function(obj.grad,experimental_relax_shapes = True)##################################################
    #if obj.verbose: print_screen(obj)#############################

    #print("Starting Adam training")
    # tf.profiler.experimental.start('../cache/tblogdir1')
    train_op_fn = train_op_inner(obj)
    #train_op_fn = train_op_inner_seperate(obj)
    epoch_base = obj.epoch_history[-1]
    #obj.u_model.save_weights('epochs_best_model')#初始化
    with trange(tf_iter) as t:
        for epoch in t:
            loss_value,loss_all = train_op_fn(obj)#,temp_list
            # Description will be displayed on the left
            t.set_description('Adam epoch %i' % (epoch + 1))
            #calculate_weight_of_loss_terms(obj)
            # Postfix will be displayed on the right,
            # formatted automatically based on argument's datatype


            if epoch % 10 == 0:
                obj.loss_history.append(loss_value)
                obj.epoch_history.append(epoch+epoch_base)
                obj.loss_all_history.append(loss_all)
                if epoch % 500 == 0:
                    loss_names = ['L_Reynolds', 'L_FB']
                    loss_str = ' | '.join([f'{name}={loss_all[i].numpy():.3e}' for i, name in enumerate(loss_names) if i < len(loss_all)])
                    print(f'  Epoch {epoch+epoch_base}: Total={loss_value.numpy():.3e} | {loss_str}')
                t.set_postfix(loss=loss_value.numpy())


                #adaptive_constant_new = obj.calculate_weight_of_loss_terms()
                #obj.adaptive_constant_func(adaptive_constant_new)
                obj.adaptive_constant_func_list.append(np.array(obj.adaptive_constant_func.adaptive_constant))

                if obj.MTL_adapt:
                   obj.MTL_adapt_list.append(np.array(obj.MTL_adapt_par[0]))
            if epoch % 100 == 0:
                if loss_value < obj.loss_value_min:
                    obj.loss_value_min = loss_value
                    #obj.save('epochs_best_model')
                    obj.u_model.save_weights(obj.best_weights_path)
            #if (epoch % 5000 == 0)&(epoch!=0)&(epoch!=tf_iter):###########
            #    obj.RAD_FB(obj.f_model_FB, obj.N_f_true, num_add_points_test=round(obj.ratio_num_RAD_FB * obj.N_f_true),
            #             num_add_points=round(obj.ratio_RAD_FB * obj.N_f_true), k=obj.k_RAD, c=obj.c_RAD)

            #obj.RAR(f_model=obj.f_model, num_add_points_test=1*len(obj.X_f_in[0]), c = obj.c_list[0])
                #obj.RAR(f_model=obj.f_model_FBNS_test, num_add_points_test=1*len(obj.X_f_in[0]), c = obj.c_list[1])
                #calculate_weight_of_loss_terms_new(obj)


    # tf.profiler.experimental.stop()

    # tf.profiler.experimental.start('../cache/tblogdir1')
    if newton_iter > 0:
        obj.n_batches = 1
        print("Starting L-BFGS training")
        if newton_eager:
            print("Executing eager-mode L-BFGS")
            loss_and_flat_grad = obj.get_loss_and_flat_grad()
            x, f_hist, currentFuncEval,loss_history,epoch_history,loss_all_history,PCGrad_COS_history,PCGrad_GMS_history=eager_lbfgs(loss_and_flat_grad,
                        get_weights(obj.u_model),
                        Struct(),obj.PCGrad_COS_GMS, maxIter=newton_iter+1, learningRate=0.8)#0.8
            obj.loss_history = obj.loss_history+loss_history
            obj.epoch_history = obj.epoch_history+list(np.array(epoch_history)+obj.epoch_history[-1])
            obj.loss_all_history = obj.loss_all_history + loss_all_history
            obj.PCGrad_COS_history = obj.PCGrad_COS_history+PCGrad_COS_history
            obj.PCGrad_GMS_history = obj.PCGrad_GMS_history+PCGrad_GMS_history
        else:
            print("Executing graph-mode L-BFGS\n Building graph...")
            print("Warning: Depending on your CPU/GPU setup, eager-mode L-BFGS may prove faster. If the computational "
                  "graph takes a long time to build, or the computation is slow, try eager-mode L-BFGS (enabled by "
                  "default)")

            obj.optim_results=lbfgs_train(obj, newton_iter)

    # tf.profiler.experimental.stop()


# @tf.function
def lbfgs_train(obj, newton_iter):
    func = graph_lbfgs(obj.u_model, obj.update_loss)

    init_params = tf.dynamic_stitch(func.idx, obj.u_model.trainable_variables)

    lbfgs_op(func, init_params, newton_iter)


@tf.function
def lbfgs_op(func, init_params, newton_iter):
    return tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function=func,
        initial_position=init_params,
        max_iterations=newton_iter,
        tolerance=1e-20,
    )#1e-20


def train_op_inner(obj):
    @tf.function(experimental_relax_shapes = True)
    def apply_grads(obj=obj):
        if obj.n_batches > 1:
            obj.batch_indx_map = np.random.choice(obj.X_f_len[0], size=obj.X_f_len[0], replace=False)

        for i in range(obj.n_batches):
            # unstack = tf.unstack(obj.u_model.trainable_variables, axis = 2)
            obj.batch = i
            #obj.variables = obj.u_model.trainable_variables
            obj.variables = obj.u_model.trainable_variables
            if obj.MTL_adapt:
                obj.variables.extend(obj.MTL_adapt_par)
                #loss_value, grads = obj.grad()
                loss_value, grads, loss_all = obj.grad_seperate_all_with_adapt_weight()
                #grads = [tf.clip_by_norm(g, 2) for g in grads]
                obj.tf_optimizer.apply_gradients(zip(grads, obj.variables))

            else:
                loss_value, grads, loss_all = obj.grad_seperate_all_with_adapt_weight()#,loss_all_orgin 实际输出是origin #,temp_list
                obj.tf_optimizer.apply_gradients(zip(grads, obj.u_model.trainable_variables))

        obj.batch = None

        return loss_value,loss_all#,temp_list

    return apply_grads


