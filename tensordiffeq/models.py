import os
import random
import tensorflow as tf
import numpy as np
import time
from .utils import *
from .networks import *
from .fit import *
from tqdm.auto import tqdm, trange
from .output import print_screen
from tensorflow import keras
from random import shuffle
import math

class CollocationSolverND:
    def __init__(self, assimilate=False, verbose=True):
        self.assimilate = assimilate
        self.verbose = verbose

    def compile(self, layer_sizes, f_model_list, domain, bcs, isAdaptive=False,
                dict_adaptive=None, init_weights=None, g=None, dist=False,
                u_model_switch=1, two_output=False, none_zero = False, adapt_True = False,
                MTL_adapt = False, PCGrad_true = False, Boundary_true = True, R_range = [],theta_range = [],
                bc_switch=1, num_freq=4, embed_dim=64):
        """
        Args:
            layer_sizes: A list of layer sizes, can be overwritten via resetting u_model to a keras model
            f_model: PDE definition
            domain: a Domain object containing the information on the domain of the system
            bcs: a list of ICs/BCs for the problem
            isAdaptive: Boolean value determining whether to implement self-adaptive solving
            dict_adaptive: a dictionary with boollean indicating adaptive loss for every loss function
            init_weights: a dictionary with keys "residual" and "BCs". Values must be a tuple with dimension
                          equal to the number of  residuals and boundares conditions, respectively
            g: a function in terms of `lambda` for self-adapting solving. Defaults to lambda^2
            dist: A boolean value determining whether the solving will be distributed across multiple GPUs

        Returns:
            None
        """
        self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=.99)
        self.tf_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=.99)
        self.tf_optimizer_weights = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=.99)
        self.layer_sizes = layer_sizes
        #self.sizes_w, self.sizes_b = get_sizes(layer_sizes)
        self.bcs = bcs

        self.f_model_list = [get_tf_model(f_model) for f_model in f_model_list]


        self.g = g
        self.domain = domain
        self.dist = dist
        self.X_f_dims = tf.shape(self.domain.X_f)
        self.X_f_len = tf.slice(self.X_f_dims, [0], [1]).numpy()
        # must explicitly cast data into tf.float32 for stability
        self.X_f_in = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]

        self.batch = None
        self.batch_indx_map = None
        self.lambdas = self.dict_adaptive = self.lambdas_map = None

        #######################################
        self.u_model_switch = u_model_switch
        if self.u_model_switch == 1:
           self.u_model = neural_net(self.layer_sizes)
        elif self.u_model_switch == 2:
           self.u_model = new_neural_net_Exact_D_BC(self.layer_sizes,self.bcs[0].val)
        elif self.u_model_switch == 3:
           self.u_model = neural_net_non_negative(self.layer_sizes)
        elif self.u_model_switch == 4:
           self.u_model = new_neural_net_Exact_D_BC_non_negative(self.layer_sizes,self.bcs[0].val)
        elif self.u_model_switch == 5:
            self.u_model = new_neural_net_two_output_3(self.layer_sizes)
        elif self.u_model_switch == 6:
            self.u_model = new_neural_net_Exact_D_BC_two_output(self.layer_sizes,self.bcs[0].val)##new_neural_net_Exact_D_BC_two_output_2
        #######################################
        elif self.u_model_switch == 8:
            self.R_range = R_range
            self.theta_range = theta_range
            self.u_model = new_neural_period_polar_exactBC_two_output(self.layer_sizes, [self.bcs[0].val, self.bcs[1].val],self.R_range,self.theta_range)

        elif self.u_model_switch == 9:
            self.R_range = R_range
            self.theta_range = theta_range
            self.u_model = new_neural_period_polar_exactBC_two_output_one(self.layer_sizes, [self.bcs[0].val, self.bcs[1].val],self.R_range,self.theta_range)

        elif self.u_model_switch == 10:
            self.R_range = R_range
            self.theta_range = theta_range
            self.u_model = new_neural_period_polar_exactBC_two_output_three(self.layer_sizes, [self.bcs[0].val, self.bcs[1].val],self.R_range,self.theta_range)

        elif self.u_model_switch == 11:
            self.R_range = R_range
            self.theta_range = theta_range
            self.u_model = new_neural_period_polar_exactBC_two_output_is_texture_func(self.layer_sizes,
                                                                            [self.bcs[0].val, self.bcs[1].val],self.R_range, self.theta_range)
        elif self.u_model_switch == 12:
            self.R_range = R_range
            self.theta_range = theta_range
            self.u_model = new_neural_period_polar_exactBC(self.layer_sizes,[self.bcs[0].val,self.bcs[1].val],self.R_range, self.theta_range)

        elif self.u_model_switch == 13:
            self.R_range = R_range
            self.theta_range = theta_range
            self.bc_switch = bc_switch
            self.num_freq = num_freq
            self.embed_dim = embed_dim
            self.u_model = new_neural_fourier_decoupled(
                self.layer_sizes,
                [self.bcs[0].val, self.bcs[1].val],
                self.R_range, self.theta_range,
                bc_switch=bc_switch, num_freq=num_freq, embed_dim=embed_dim
            )

        self.PCGrad_true = PCGrad_true#True
        self.Boundary_true = Boundary_true
        #######################################
        self.loss_history = []
        self.loss_all_history = []
        self.epoch_history = [0]
        self.adapt_True = adapt_True
        self.balance = False
        #self.adaptive_constant_list = []
        self.two_output = two_output
        self.none_zero = none_zero

        #self.g_value = Compute_g_value(len(self.X_f_in[0]))
        #self.adaptive_constant = tf.constant(np.array([100]*(len(self.update_loss_seperate())-1)), tf.float32)  # loss权重系数
        #self.adaptive_constant = tf.constant(np.array([1, 1, 1, 1]),tf.float32)  # loss权重系数
        self.adaptive_constant_alpha = tf.constant(0.2, tf.float32)
        self.adaptive_constant_func = ComputeSum_weight(len(self.update_loss_seperate()), self.adaptive_constant_alpha)

        self.adaptive_constant_alpha_PCGrad_loss = tf.constant(1e-1, tf.float32)
        self.adaptive_constant_func_PCGrad_loss = ComputeSum_weight(len(self.update_loss_seperate()), self.adaptive_constant_alpha_PCGrad_loss)

        self.adaptive_constant_func_list = []
        self.loss_value_min = 1.e12
        self.best_weights_path = 'epochs_best_model.weights.h5'  # 默认值，可被外部覆盖
        self.u_model.save_weights(self.best_weights_path)  # 初始
        #######################################
        n_x, n_y = (101, 101)
        x_point = np.linspace(-0.5, 0.5, n_x)
        y_point = np.linspace(-0.5, 0.5, n_y)
        self.X, self.Y = np.meshgrid(x_point, y_point)
        self.X_Y_star = np.hstack((self.X.flatten()[:, None], self.Y.flatten()[:, None]))
        #u_pred_temp = self.u_model(X_Y_star)
        #self.U_pred_temp = tdq.plotting.get_griddata(X_Y_star, u_pred_temp.numpy().reshape(1, -1).flatten(), (self.X, self.Y))
        #self.U_pred_temp = self.U_pred_temp_old

        '''result_comsol = np.loadtxt('p_Reynold.txt')
        U_comsol =  result_comsol[:, 2].flatten()
        self.U_comsol_flatten = U_comsol.flatten()
        self.non_nan_list = []
        for i in range(len(self.U_comsol_flatten)):
            tem_value = self.U_comsol_flatten[i]
            if math.isnan(tem_value):
                continue
            else:
                self.non_nan_list.append(i)'''

        self.u_pred_cal_old = self.u_model(self.domain.X_f)
        self.error_cal = []
        self.error_L2_list = []
        #######################################
        self.MTL_adapt = MTL_adapt
        if self.MTL_adapt:
            self.MTL_adapt_par = [(tf.Variable([1.]*len(self.update_loss_seperate()), trainable=True, dtype=tf.float32))]#,constraint=keras.constraints.NonNeg()
            self.MTL_adapt_list = []
        #######################################
        '''self.layer_trainable_variables_num = [int(tf.reduce_sum([tf.reduce_sum(trainable_variables_value * 0 + 1) for trainable_variables_value in trainable_variables])) for
                      trainable_variables in self.u_model.trainable_variables]  # 除res外'''

        ##############版本问题###############################
        self.sizes_w = [int(tf.reduce_sum(trainable_variables * 0 + 1))
                        if 'kernel' in trainable_variables.name else 0 for trainable_variables in self.u_model.trainable_variables]  # 除res外
        self.sizes_b = [int(tf.reduce_sum(trainable_variables * 0 + 1))
                        if 'bias' in trainable_variables.name else 0 for trainable_variables in self.u_model.trainable_variables]  # 除res外

        #del self.sizes_b[0]
        self.sizes_b = list(filter(lambda x: x != 0, self.sizes_b))
        self.sizes_w = list(filter(lambda x: x != 0, self.sizes_w))
        #self.sizes_b.append(0)
        #######################################


    def compile_data(self, x, t, y):
        if not self.assimilate:
            raise Exception(
                "Assimilate needs to be set to 'true' for data assimilation. Re-initialize CollocationSolver1D with "
                "assimilate=True.")
        self.data_x = x
        self.data_t = t
        self.data_s = y

    def update_loss(self):
        loss_all = []

        # Residual Equations
        #####################################
        loss_res = self.update_loss_res()
        loss_all.append(loss_res)

        # boundary condition
        if (self.Boundary_true==True):
           loss_bcs = self.update_loss_bcs()
           loss_all = loss_all + loss_bcs#每个边界分开储存

        #####################################第二值的边界条件############################################
        #####################################一值和二值的关联############################################
        if (self.two_output==True):
           loss_g_all = self.update_loss_JFO_term_interact()
           loss_bcs_2 = self.update_loss_JFO_second_term_BCs()
           loss_all.append(loss_g_all)
           loss_all = loss_all + loss_bcs_2

        #雷诺边界条件#这个方法效果不好
        if self.none_zero ==True:
           loss_zero_all = self.update_loss_0_boundary()
           #loss_zero_all = self.zero_para * loss_zero_all
           loss_all.append(loss_zero_all)
        self.loss_all = loss_all
        #loss_total = 1*loss_res + tf.reduce_sum(tf.math.multiply(loss_all[1:], self.adaptive_constant))#self.adaptive_constant
        #loss_total = 1 * loss_res + tf.reduce_sum(tf.math.multiply(loss_all[1:], self.adaptive_constant_func.adaptive_constant))  # self.adaptive_constant
        #loss_total = tf.reduce_sum(tf.math.multiply(loss_all, self.adaptive_constant_func.adaptive_constant))  # self.adaptive_constant
        loss_total = self.adaptive_constant_func.adaptive_constant[0][0] * loss_all[0] \
                     + tf.reduce_sum(tf.math.multiply(loss_all[1:], self.adaptive_constant_func.adaptive_constant[0][1:]))
        return loss_total,loss_all #lbfgs-graph ,后面输出省略，否则不省略


    def update_loss_seperate(self):
        loss_all = []
        #####################################
        for f_model in self.f_model_list:
            loss_res = self.update_loss_res(f_model)#这个输出本身就是list
            #loss_all.append(loss_res)
            loss_all = loss_all + loss_res

        # boundary condition
        if (self.Boundary_true==True):
           loss_bcs = self.update_loss_bcs()
           loss_all = loss_all + loss_bcs#每个边界分开储存

        #####################################第二值的边界条件############################################
        #####################################一值和二值的关联############################################
        if (self.two_output==True):
           loss_g_all = self.update_loss_JFO_term_interact()
           loss_all.append(loss_g_all)
           #loss_g_all_2 = self.update_loss_JFO_term_interact_2()
           #loss_all.append(loss_g_all_2)
           #loss_bcs_2 = self.update_loss_JFO_second_term_BCs()
           #loss_all = loss_all + loss_bcs_2


        #雷诺边界条件
        if self.none_zero == True:
           loss_zero_all = self.update_loss_0_boundary()
           #loss_zero_all = self.zero_para * loss_zero_all
           loss_all.append(loss_zero_all)
        return loss_all#loss_total#,

    def update_loss_res(self,f_model):
        #####################################
        # Residual Equations
        #####################################
        # pass thorough the forward method

        f_u_preds = f_model(self.u_model, *self.X_f_in)

        # If it is only one residual, just convert it to a tuple of one element
        if not isinstance(f_u_preds, tuple):
            f_u_preds = f_u_preds,

        loss_res = []

        for counter_res, f_u_pred in enumerate(f_u_preds):
            # Check if the current Residual is adaptive


            loss_r = MSE(f_u_pred, constant(0.0))#这个也是平均过的，输入值一次进去

            #self.counter_res = counter_res
            #loss_res = tf.math.add(loss_r, loss_res)
            loss_res.append(loss_r)

        return loss_res

    def update_loss_bcs(self):
        loss_bcs = []
        #loss_bcs = 0.
        #####################################
        # BOUNDARIES and INIT conditions
        #####################################
        # Check if adaptive is allowed

            #idx_lambda_bcs = 0

        for counter_bc, bc in enumerate(self.bcs):#一个边界条件一个边界条件来
            # Check if the current BS is adaptive


            isBC_adaptive = False
            loss_bc = 0.
            # Periodic BC iteration for all components of deriv_model
            if bc.isPeriodic:
                for i, dim in enumerate(bc.var):
                        for j, lst in enumerate(dim):
                            for k, tup in enumerate(lst):
                                upper = bc.u_x_model(self.u_model, bc.upper[i])[j][k]#点的值
                                lower = bc.u_x_model(self.u_model, bc.lower[i])[j][k]#点的值
                                msq = MSE(upper, lower)
                                loss_bc = tf.math.add(loss_bc, msq)#################
            # initial BCs, including adaptive model
            elif bc.isInit:
                loss_bc = MSE(self.u_model(bc.input), bc.val)
            # BC types are added
            elif bc.isNeumann:

                for i, dim in enumerate(bc.var):
                        for j, lst in enumerate(dim):
                            for k, tup in enumerate(lst):
                                target = tf.cast(bc.u_x_model(self.u_model, bc.input[i])[j][k], dtype=tf.float32)
                                msq = MSE(bc.val, target)
                                loss_bc = tf.math.add(loss_bc, msq)

            elif bc.isDirichlect:

                if (self.two_output == True):
                    #self.bc_loss_temp = self.u_model(bc.input)[0]
                    loss_bc = MSE(tf.reshape(self.u_model(bc.input)[0],[-1,1]), bc.val)#均值过的
                else:
                    loss_bc = MSE(tf.reshape(self.u_model(bc.input), [-1, 1]), bc.val)  # 均值过的
            else:
                raise Exception('Boundary condition type is not acceptable')
            loss_bcs.append(loss_bc)#每条边界都记录
        return [tf.reduce_mean(loss_bcs)]#[tf.reduce_sum(loss_bcs)]#loss_bcs

    def update_loss_JFO_term_interact(self):
        loss_g_all = 0.
        u_preds = self.u_model(self.domain.X_f)

        # If it is only one residual, just convert it to a tuple of one element
        if not isinstance(u_preds, tuple):
            u_preds = u_preds,
        for counter_JFO, u_pred in enumerate(u_preds):
            #self.g_value(tf.math.sign(u_pred[0]) / 2 + 1 / 2)
            loss_g = MSE(u_pred[0]+u_pred[1]-tf.math.sqrt((u_pred[0]**2+u_pred[1]**2)), constant(0.0))
            #K_BIG = 1.;
            #loss_g = MSE(K_BIG*u_pred[0] + u_pred[1] - tf.math.sqrt(((K_BIG*u_pred[0]) ** 2 + u_pred[1] ** 2)),constant(0.0))
            #loss_g = MSE(tf.nn.relu(0.870 - tf.reduce_max(u_pred[0])),constant(0.0))
            loss_g_all = tf.math.add(loss_g, loss_g_all)

        return loss_g_all

    def update_loss_JFO_term_interact_2(self):
        loss_g_all = 0.
        u_preds = self.u_model(self.domain.X_f)

        # If it is only one residual, just convert it to a tuple of one element
        if not isinstance(u_preds, tuple):
            u_preds = u_preds,
        for counter_JFO, u_pred in enumerate(u_preds):
            #self.g_value(tf.math.sign(u_pred[0]) / 2 + 1 / 2)
            #loss_g = MSE(u_pred[0]+u_pred[1]-tf.math.sqrt((u_pred[0]**2+u_pred[1]**2))+u_pred[0]*u_pred[1], constant(0.0))
            K_BIG = 1.;
            #loss_g = MSE(K_BIG*u_pred[0] + u_pred[1] - tf.math.sqrt(((K_BIG*u_pred[0]) ** 2 + u_pred[1] ** 2)),constant(0.0))
            loss_g = MSE(tf.nn.relu(tf.reduce_min(u_pred[0])-0.),constant(0.0))
            loss_g_all = tf.math.add(loss_g, loss_g_all)

        return loss_g_all

    def update_loss_JFO_second_term_BCs(self):
        loss_bcs_2 = []
        #loss_bcs_2 = 0.
        for counter_bc, bc in enumerate(self.bcs):  # 一个边界条件一个边界条件来

            if bc.isDirichlect:
                    loss_bc_2 = MSE(tf.reshape(self.u_model(bc.input)[1], [-1, 1]),0)###1,0  # 这个是均匀过的###################################
                    #loss_bc_2 = -   tf.stop_gradient(tf.math.sign(tf.reshape(self.u_model(bc.input)[0], [-1, 1])) / 2 + 1 / 2) * tf.math.log(tf.reshape(self.u_model(bc.input)[1], [-1, 1])) \
                             #- (1 - tf.stop_gradient(tf.math.sign(tf.reshape(self.u_model(bc.input)[0], [-1, 1])) / 2 + 1 / 2)) * tf.math.log(1 - tf.reshape(self.u_model(bc.input)[1], [-1, 1]))
            else:
                raise Exception('Boundary condition type is not acceptable')
            loss_bcs_2.append(loss_bc_2)

        return [tf.reduce_mean(loss_bcs_2)]#loss_bcs_2#

    def update_loss_0_boundary(self):
        u_preds = self.u_model(self.domain.X_f)
        # If it is only one residual, just convert it to a tuple of one element
        if not isinstance(u_preds, tuple):
            u_preds = u_preds,

        loss_zero_all = 0.
        for counter_zero, u_pred in enumerate(u_preds):


            loss_zero = tf.reduce_mean(tf.math.add(tf.math.multiply(tf.math.abs(u_pred),-u_pred),tf.math.pow(u_pred,2))/2)#
                #loss_zero = MSE(tf.nn.relu(-u_pred), constant(0.0))
            # Check if the current Residual is adaptive
            loss_zero_all = tf.math.add(loss_zero_all, loss_zero)
        return loss_zero_all


    def update_loss_origin(self):
        loss_bcs = 0.

        #####################################
        # BOUNDARIES and INIT conditions
        #####################################
        # Check if adaptive is allowed
        if self.isAdaptive:
            if len(self.lambdas_map['bcs']) > 0:
                idx_lambda_bcs = self.lambdas_map['bcs'][0]

        for counter_bc, bc in enumerate(self.bcs):
            loss_bc = 0.
            # Check if the current BC is adaptive
            if self.isAdaptive:
                isBC_adaptive = self.dict_adaptive["BCs"][counter_bc]
            else:
                isBC_adaptive = False

            # Periodic BC iteration for all components of deriv_model
            if bc.isPeriodic:
                if isBC_adaptive:
                    # TODO: include Adapative Periodic Boundaries Conditions
                    raise Exception('TensorDiffEq is currently not accepting Adapative Periodic Boundaries Conditions')
                else:
                    for i, dim in enumerate(bc.var):
                        for j, lst in enumerate(dim):
                            for k, tup in enumerate(lst):
                                upper = bc.u_x_model(self.u_model, bc.upper[i])[j][k]
                                lower = bc.u_x_model(self.u_model, bc.lower[i])[j][k]
                                msq = MSE(upper, lower)
                                loss_bc = tf.math.add(loss_bc, msq)
            # initial BCs, including adaptive model
            elif bc.isInit:
                if isBC_adaptive:
                    loss_bc = MSE(self.u_model(bc.input), bc.val, self.lambdas[idx_lambda_bcs])
                    idx_lambda_bcs += 1
                else:
                    loss_bc = MSE(self.u_model(bc.input), bc.val)
            # BC types are added
            elif bc.isNeumann:
                if isBC_adaptive:
                    #TODO: include Adapative Neumann Boundaries Conditions
                    raise Exception('TensorDiffEq is currently not accepting Adapative Neumann Boundaries Conditions')
                else:
                    for i, dim in enumerate(bc.var):
                        for j, lst in enumerate(dim):
                            for k, tup in enumerate(lst):
                                target = tf.cast(bc.u_x_model(self.u_model, bc.input[i])[j][k], dtype=tf.float32)
                                msq = MSE(bc.val, target)
                                loss_bc = tf.math.add(loss_bc, msq)

            elif bc.isDirichlect:
                if isBC_adaptive:
                    loss_bc = MSE(self.u_model(bc.input), bc.val, self.lambdas[idx_lambda_bcs])
                    idx_lambda_bcs += 1
                else:
                    loss_bc = MSE(self.u_model(bc.input), bc.val)

            else:
                raise Exception('Boundary condition type is not acceptable')

            loss_bcs = tf.add(loss_bcs, loss_bc)

        #####################################
        # Residual Equations
        #####################################
        # pass thorough the forward method
        if self.n_batches > 1:
            # The collocation points will be split based on the batch_indx_map
            # generated on the beginning of this epoch on models.train_op_inner.apply_grads
            X_batch = []
            for x_in in self.X_f_in:
                indx_on_batch = self.batch_indx_map[self.batch * self.batch_sz:(self.batch + 1) * self.batch_sz]
                X_batch.append(tf.gather(x_in,indx_on_batch))
            f_u_preds = self.f_model(self.u_model, *X_batch)
        else:
            f_u_preds = self.f_model(self.u_model, *self.X_f_in)

        # If it is only one residual, just convert it to a tuple of one element
        if not isinstance(f_u_preds, tuple):
            f_u_preds = f_u_preds,

        loss_res = 0.
        for counter_res, f_u_pred in enumerate(f_u_preds):
            # Check if the current Residual is adaptive
            if self.isAdaptive:
                isRes_adaptive = self.dict_adaptive["residual"][counter_res]
                if isRes_adaptive:
                    idx_lambda_res = self.lambdas_map['residual'][0]
                    lambdas2loss = self.lambdas[idx_lambda_res]

                    if self.n_batches > 1:
                        # select lambdas on minebatch
                        lambdas2loss = tf.gather(lambdas2loss,indx_on_batch)

                    if self.g is not None:
                        loss_r = g_MSE(f_u_pred, constant(0.0), self.g(lambdas2loss))
                    else:
                        loss_r = MSE(f_u_pred, constant(0.0), lambdas2loss)
                    idx_lambda_res += 1
                else:
                    # In the case where the model is Adaptive but the residual
                    # is not adaptive, the residual loss should be computed.
                    loss_r = MSE(f_u_pred, constant(0.0))
            else:
                loss_r = MSE(f_u_pred, constant(0.0))

            loss_res = tf.math.add(loss_r, loss_res)

        loss_total = tf.math.add(loss_res, loss_bcs)

        return loss_total

    # @tf.function
    def grad(self):
        with tf.GradientTape() as tape:
            loss_value = self.update_loss()
        grads = tape.gradient(loss_value, self.variables)
        return loss_value, grads

    # @tf.function
    def grad_seperate_all_with_adapt_weight(self):#对神经网络系数进行求导
        with tf.GradientTape(persistent=True) as tape:
            loss_all = self.update_loss_seperate()#每项分别取出来
            #loss_all_origin = loss_all
            #loss_total = self.update_loss()  # 每项分别取出来
            if self.MTL_adapt:
                loss_total = tf.reduce_sum(tf.math.multiply(loss_all[0:], 1 / 2 * tf.math.exp(-self.MTL_adapt_par[0][0:]))) + tf.math.reduce_sum(self.MTL_adapt_par[0][0:])
            else:
                loss_total = tf.reduce_sum(tf.math.multiply(loss_all[0:], self.adaptive_constant_func.adaptive_constant[0][0:]))

        grad_total = tape.gradient(loss_total, self.variables)

        if self.adapt_True :
            grad_all = [tape.gradient(loss_bc, self.variables) for loss_bc in loss_all]
            grad_all_withoutnone = [list(filter(lambda x: x != None, grad_w)) for grad_w in grad_all]  # 去除none
            grad_all_w = [grad_w[0::2] for grad_w in grad_all_withoutnone]  # 取出系数w

            #grads_res_max = tf.reduce_max([tf.reduce_max(tf.abs(grads_res_w_layer)) for grads_res_w_layer in grad_all_w[0]])

            grad_res_num = [
                tf.reduce_sum([tf.reduce_sum(grad_layer_w * 0 + 1) for grad_layer_w in grad_all_w[0]])]  #
            grad_res_list = [
                tf.reduce_sum([tf.reduce_sum(tf.abs(grad_layer_w)) for grad_layer_w in grad_all_w[0]])]  #
            grads_res_mean = tf.math.divide(grad_res_list, grad_res_num)

            grad_w_num = [tf.reduce_sum([tf.reduce_sum(grad_layer_w * 0 + 1) for grad_layer_w in grad_all_w_term]) for
                          grad_all_w_term in grad_all_w[0:]]  # 除res外

            grad_w_list = [tf.reduce_sum([tf.reduce_sum(tf.abs(grad_layer_w)) for grad_layer_w in grad_all_w_term]) for
                           grad_all_w_term in grad_all_w[0:]]  # 除res外
            grads_mean_list = tf.math.divide(grad_w_list, grad_w_num) + 0.000000000001
            #adaptive_constant_new = tf.math.divide(grads_res_max, grads_mean_list)
            adaptive_constant_new = tf.math.divide(grads_res_mean, grads_mean_list)
            adaptive_constant_new = tf.minimum(adaptive_constant_new, 1e12)#1e12
            adaptive_constant_new = tf.maximum(adaptive_constant_new, 1e-2)
            self.adaptive_constant_func(adaptive_constant_new)  # 更新系数
            # self.adaptive_constant_func_list.append(self.adaptive_constant_func.adaptive_constant)
        if self.PCGrad_true:
            #tf.random.shuffle(loss_all)
            grad_all = [tape.gradient(loss, self.variables) for loss in loss_all]

            num_tasks = len(loss_all)


            for counter_grad_all in range(num_tasks):
               for counter_grad, grad in enumerate(grad_all[counter_grad_all]):
                   if grad == None:
                       grad_all[counter_grad_all][counter_grad] = 0.*grad_total[counter_grad]
            #grads_task = [tf.concat([tf.reshape(grad, [-1, ]) for grad in grads], axis=0) for grads in grad_all]#grad_all_flatten
            grads_task = [tf.concat([tf.reshape(grad, [-1, ]) for grad in grads], axis=0) for grads in grad_all]
            if self.balance:
                self.adaptive_constant_func_PCGrad_loss(tf.convert_to_tensor(loss_all))
                loss_all_smooth = self.adaptive_constant_func_PCGrad_loss.adaptive_constant
                loss_all_smooth_reference = self.adaptive_constant_func_PCGrad_loss.adaptive_constant_step
                loss_effective = [loss_all_smooth[:, k] / loss_all_smooth_reference[:, k] for k in range(num_tasks)]
                loss_effective_01 = [loss_effective[k] / tf.reduce_sum(loss_effective) for k in range(num_tasks)]

                grads_task_norm = [tf.math.sqrt(tf.reduce_sum(grads**2)) for grads in grads_task]

                #weight_grads_task_norm = [tf.reduce_mean(grads_task_norm)/grads_norm for grads_norm in grads_task_norm]
                weight_grads_task_norm = [(loss_effective_01[k]*(tf.reduce_mean(grads_task_norm)-grads_task_norm[k])+grads_task_norm[k]) / grads_task_norm[k] for k in range(num_tasks)]

                grads_task_weighted =  [grads_task[k]* weight_grads_task_norm[k] for k in range(num_tasks)]

                grads_task = grads_task_weighted

            np.random.shuffle(grads_task)
            grads_task_proj = []
            for grad_task in (grads_task):
                for k in range(num_tasks):
                    inner_product = tf.reduce_sum(grad_task * grads_task[k])
                    proj_direction = inner_product / tf.reduce_sum(grads_task[k] * grads_task[k]+1e-12)#防止当一个任务梯度为0时，出现NaN
                    grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
                grads_task_proj.append(grad_task)
            #self.grads_task_proj = grads_task_proj
            proj_grads_flatten = grads_task_proj


            #重新组合成各层的形式并求和
            proj_grads = []
            for j in range(num_tasks):
                start_idx = 0
                for idx, var in enumerate(self.variables):
                    grad_shape = var.get_shape()
                    flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                    proj_grad = proj_grads_flatten[j][start_idx:start_idx + flatten_dim]
                    proj_grad = tf.reshape(proj_grad, grad_shape)
                    if len(proj_grads) < len(self.variables):
                        proj_grads.append(proj_grad)
                    else:
                        proj_grads[idx] += proj_grad#直接求和了
                    start_idx += flatten_dim
            grad_total = proj_grads
            loss_total = tf.reduce_sum(loss_all)
        return loss_total, grad_total, loss_all#loss_all# , grad_net_1,grad_net_2



    def fit(self, tf_iter=0, newton_iter=0, batch_sz=None, newton_eager=True):

        # Can adjust batch size for collocation points, here we set it to N_f
        N_f = self.X_f_len[0]
        self.batch_sz = batch_sz if batch_sz is not None else N_f
        self.n_batches = N_f // self.batch_sz


        if self.n_batches > 1 and self.dist:
            raise Exception("Currently we dont support distributed minibatching training")

        if self.dist:
            BUFFER_SIZE = len(self.X_f_in[0])
            EPOCHS = tf_iter
            # devices = ['/gpu:0', '/gpu:1','/gpu:2', '/gpu:3'],
            try:
                self.strategy = tf.distribute.MirroredStrategy()
            except:
                print(
                    "Looks like we cant find any GPUs available, or your GPUs arent responding to Tensorflow's API. If "
                    "you're receiving this in error, check that your CUDA, "
                    "CUDNN, and other GPU dependencies are installed correctly with correct versioning based on your "
                    "version of Tensorflow")

            print("Number of GPU devices: {}".format(self.strategy.num_replicas_in_sync))

            BATCH_SIZE_PER_REPLICA = self.batch_sz
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync

            # options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

            self.train_dataset = tf.data.Dataset.from_tensor_slices(
                self.X_f_in).batch(GLOBAL_BATCH_SIZE)

            # self.train_dataset = self.train_dataset.with_options(options)

            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)

            start_time = time.time()

            with self.strategy.scope():
                self.u_model = neural_net(self.layer_sizes)
                self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=.99)
                self.tf_optimizer_weights = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=.99)
                # self.dist_col_weights = tf.Variable(tf.zeros(batch_sz), validate_shape=True)



            fit_dist(self, tf_iter=tf_iter, newton_iter=newton_iter, batch_sz=batch_sz, newton_eager=newton_eager)

        else:
            fit(self, tf_iter=tf_iter, newton_iter=newton_iter, newton_eager=newton_eager)

    # L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
    def get_loss_and_flat_grad(self):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                set_weights(self.u_model, w, self.sizes_w, self.sizes_b)
                loss_value, loss_all = self.update_loss()#

            grad = tape.gradient(loss_value, self.u_model.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat , loss_all

        return loss_and_flat_grad



    def predict(self, X_star):
        # predict using concatenated data
        u_star = self.u_model(X_star)
        # split data into tuples for ND support
        # must explicitly cast data into tf.float32 for stability
        # tmp = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        # X_star = np.asarray(tmp)
        # X_star = tuple(X_star)
        X_star = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        f_u_star = [f_model(self.u_model, *X_star) for f_model in self.f_model_list]
        return u_star, f_u_star

    def predict_test(self, X_star):
        # predict using concatenated data
        u_star = self.u_model(X_star)
        # split data into tuples for ND support
        # must explicitly cast data into tf.float32 for stability
        # tmp = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        # X_star = np.asarray(tmp)
        # X_star = tuple(X_star)
        X_star = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_star.T)]
        f_u_star = self.f_model(self.u_model, *X_star)
        return u_star, f_u_star

    def save(self, path):
        self.u_model.save(path)

    def load_model(self, path, compile_model=False):
        self.u_model = tf.keras.models.load_model(path, compile=compile_model)

    def calculate_weight_of_loss_terms(self):
        loss_all, grad_all = self.grad_seperate_all()
        grad_all_withoutnone = [list(filter(lambda x: x != None, grad_w)) for grad_w in grad_all]  # 去除none
        grad_all_w = [grad_w[0::2] for grad_w in grad_all_withoutnone]  # 取出系数w

        #grads_res_max = tf.reduce_max([tf.reduce_max(tf.abs(grads_res_w_layer)) for grads_res_w_layer in grad_all_w[0]])

        grad_res_num = [tf.reduce_sum([tf.reduce_sum(grad_layer_w * 0 + 1) for grad_layer_w in grad_all_w[0]])]  # 除res外
        grad_res_list = [tf.reduce_sum([tf.reduce_sum(tf.abs(grad_layer_w)) for grad_layer_w in grad_all_w[0]])]  # 除res外
        grads_res_mean = tf.math.divide(grad_res_list, grad_res_num)

        grad_w_num = [tf.reduce_sum([tf.reduce_sum(grad_layer_w * 0 + 1) for grad_layer_w in grad_all_w_term]) for
                      grad_all_w_term in grad_all_w[1:]]  # 除res外

        grad_w_list = [tf.reduce_sum([tf.reduce_sum(tf.abs(grad_layer_w)) for grad_layer_w in grad_all_w_term]) for
                       grad_all_w_term in grad_all_w[1:]]  # 除res外
        grads_mean_list = tf.math.divide(grad_w_list, grad_w_num) + 0.000000000001
        #self.grads_res_max = grads_res_max
        #self. grads_mean_list =  grads_mean_list
        adaptive_constant_new = tf.math.divide(grads_res_mean, grads_mean_list)#grads_res_max
        # adaptive_constant_new = tf.nn.relu6(tf.math.divide(grads_res_max, grads_mean_list)/1e8)/6*1e8
        return adaptive_constant_new
        # obj.adaptive_constant = (1 - obj.adaptive_constant_alpha)*obj.adaptive_constant + obj.adaptive_constant_alpha * adaptive_constant_new
        #self.adaptive_constant = tf.math.multiply((1.0 - self.adaptive_constant_alpha),self.adaptive_constant) + tf.math.multiply(self.adaptive_constant_alpha,adaptive_constant_new)
        #self.adaptive_constant_list.append(self.adaptive_constant)


    def calculate_weight_of_loss_terms_simple(self):
        loss_all = self.update_loss_seperate()

        #self.grads_res_max = grads_res_max
        #self. grads_mean_list =  grads_mean_list
        adaptive_constant_new = tf.math.divide(loss_all[0], loss_all[1:])
        # adaptive_constant_new = tf.nn.relu6(tf.math.divide(grads_res_max, grads_mean_list)/1e8)/6*1e8
        return adaptive_constant_new



    def RAR(self,f_model,num_add_points_test=2000,c = 0.8):
        #num_add_points = 500

        #X_star = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]
        X_star = self.X_f_in

        f_u_1_pred = f_model(self.u_model, *X_star)
        #f_u_1_pred = self.f_model(self.u_model, *X_star)
        #f_u_2_pred = self.f_model_FBNS_test(self.u_model, *X_star)

        X_f_in = self.domain.generate_collocation_points_old(num_add_points_test)

        X,Y = X_f_in[:,0],X_f_in[:,1]

        f_u_1_pred_interpolate = griddata(self.domain.X_f, np.array(f_u_1_pred).flatten(), (X, Y), method='cubic')

        f_u_1_pred_interpolate_nonnan = np.delete(f_u_1_pred_interpolate, np.where(np.isnan(f_u_1_pred_interpolate)))
        X_f_in_nonnan = X_f_in[np.where(1-np.isnan(f_u_1_pred_interpolate)),:][0,:,:]

        f_u_1_pred_interpolate_nonnan = (f_u_1_pred_interpolate_nonnan)**2
        #f_u_2_pred_interpolate = griddata(self.domain.X_f, np.array(f_u_2_pred).flatten(), (X, Y), method='cubic')

        point_with_value = np.concatenate([X_f_in_nonnan, np.reshape(f_u_1_pred_interpolate_nonnan,[-1,1])], axis=1)
        #point_with_value_sorted = point_with_value[point_with_value[:, 2].argsort()][:,[0,1]]#按照第三列对行进行排序,递增
        #value_sorted = point_with_value[point_with_value[:, 2].argsort()][:, 2]
        #index_points = value_sorted.index(next(x for x in value_sorted if x > c * value_sorted[-1]))
        point_add = point_with_value[point_with_value[:,2]>c*np.max(point_with_value[:,2]),0:2]

        add_x = tf.reshape(tf.constant(point_add[:,0], dtype=tf.float32),[-1,1])
        add_y = tf.reshape(tf.constant(point_add[:,1], dtype=tf.float32),[-1,1])

        self.domain.X_f = np.concatenate((self.domain.X_f, tf.concat([add_x, add_y], 1)), 0)
        #self.X_f_in = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]
        self.X_f_in = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]

    def RAD_FB(self,f_model_list,N_raw,num_add_points_test=2500,num_add_points=[50],k = 1.0, c = 1.0):

        self.domain.X_f = self.domain.X_f[0:N_raw, :]
        X_f_in = self.domain.generate_collocation_points_old(num_add_points_test)
        X_star = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(X_f_in.T)]

        for i in range(len(f_model_list)):
           f_model = f_model_list[i]
           f_u_1_pred_interpolate = f_model(self.u_model, *X_star)

           #f_u_1_pred_interpolate_nonnan = f_u_1_pred_interpolate[1-np.isnan(f_u_1_pred_interpolate)]
           #X_f_in_nonnan = X_f_in[1-np.isnan(f_u_1_pred_interpolate),:]
           f_u_1_pred_interpolate_nonnan = np.delete(f_u_1_pred_interpolate, np.where(np.isnan(f_u_1_pred_interpolate)))
           X_f_in_nonnan = X_f_in[np.where(1-np.isnan(f_u_1_pred_interpolate)),:][0,:,:]


           f_u_1_pred_interpolate_nonnan = (f_u_1_pred_interpolate_nonnan)**2
           #f_u_2_pred_interpolate = griddata(self.domain.X_f, np.array(f_u_2_pred).flatten(), (X, Y), method='cubic')


           err_eq = np.power(f_u_1_pred_interpolate_nonnan, k) / (np.power(f_u_1_pred_interpolate_nonnan, k).mean()+1e-8) + c
           err_eq[np.isnan(err_eq)]=0
           err_eq_normalized = (err_eq / sum(err_eq))#[:, 0]
           X_ids = np.random.choice(a=len(X_f_in_nonnan), size=num_add_points[i], replace=False, p=err_eq_normalized)

           X_f_in_add = X_f_in_nonnan[X_ids,:]
           #X_selected = X[X_ids]
           #Y_selected = Y[X_ids]

           #add_x = tf.reshape(tf.constant(X[X_ids], dtype=tf.float32),[-1,1])
           #add_y = tf.reshape(tf.constant(Y[X_ids], dtype=tf.float32),[-1,1])


           self.domain.X_f = np.concatenate((self.domain.X_f, X_f_in_add), 0)
        #self.domain.X_f = np.concatenate((self.domain.X_f, tf.concat([add_x, add_y], 1)), 0)

        #self.domain.X_f = tf.concat([add_x, add_y], 1) #直接替换

        self.X_f_in = [tf.cast(np.reshape(vec, (-1, 1)), tf.float32) for i, vec in enumerate(self.domain.X_f.T)]

class ComputeSum_weight(keras.layers.Layer):
        def __init__(self, input_dim, adaptive_constant_alpha, adaptive_constant_step_alpha=100):
            super(ComputeSum_weight, self).__init__()
            #self.adaptive_constant = tf.Variable(initial_value=tf.ones((1, input_dim)), trainable=False)#更新系数而非保留系数
            self.adaptive_constant = tf.Variable(initial_value=tf.reshape(tf.constant([1.]*input_dim,dtype=tf.float32),(1,input_dim)), trainable=False)  # 更新系数而非保留系数
            self.adaptive_constant_alpha = tf.Variable(adaptive_constant_alpha, trainable=False, dtype=tf.float32)
            #self.count = tf.Variable(initial_value=tf.ones(1), trainable=False)
            #self.one = tf.Variable(initial_value=tf.ones(1), trainable=False)

            self.adaptive_constant_step = tf.Variable(initial_value=tf.reshape(tf.constant([1.] * input_dim, dtype=tf.float32), (1, input_dim)),trainable=False)  # 更新系数而非保留系数
            self.adaptive_constant_step_alpha = adaptive_constant_step_alpha
            self.count = self.adaptive_constant_step_alpha
        def call(self, adaptive_constant_new):

            self.adaptive_constant.assign(tf.math.multiply((1.0 - self.adaptive_constant_alpha), self.adaptive_constant) +
                                      tf.math.multiply(self.adaptive_constant_alpha, adaptive_constant_new))
            self.count +=1
            if (self.count-self.adaptive_constant_step_alpha)>0:
                self.adaptive_constant_step.assign(self.adaptive_constant)
                self.count = 0
            #self.count.assign(self.count+self.one)

