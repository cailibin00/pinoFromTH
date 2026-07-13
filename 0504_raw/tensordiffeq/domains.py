import numpy as np
from .utils import LatinHypercubeSample


class DomainND:
    def __init__(self, var, time_var=None):
        self.vars = var
        self.domaindict = []
        self.domain_ids = []
        self.time_var = time_var

    def generate_collocation_points_old(self, N_f):
        range_list = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]

        limits = np.array(range_list)  # x,t domain
        X_f = LatinHypercubeSample(N_f, limits)

        #range_list[0][0], range_list[0][1] = -1 * 0.2 + 1/2, - (0 - 1) * 0.2 + 1/2
        #limits_middle = np.array(range_list)
        #X_f_4 = LatinHypercubeSample(round(N_f*0.5), limits_middle)
        #X_f = np.concatenate((X_f, X_f_4), axis=0)  # 补充
        return X_f
        #self.X_f = X_f

    def generate_collocation_points(self, N_f,points_scala):
        range_list = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]
        range_list_2 = [
            [val for key, val in dict_.items() if "range" in key][0]
            for dict_ in self.domaindict
        ]
        limits = np.array(range_list)  # x,t domain
        limits_out = np.array(range_list)  # x,t domain
        #limits_middle = np.array(range_list)  # x,t domain

        # X_f = LatinHypercubeSample(N_f, limits)

        limits_out_part = limits_out
        #limits_out_part[0][0] = 0.8 * (limits_out_part[0][1] - limits_out_part[0][0]) + limits_out_part[0][0]
        # X_f_1 = LatinHypercubeSample(round(N_f), limits)
        X_f_1 = LatinHypercubeSample(round(N_f), limits_out_part)

        #limits_middle_part = limits_middle

        '''limits_middle_part[0][1] = 0.8 * (limits_middle_part[0][1] - limits_middle_part[0][0]) + limits_middle_part[0][0]
        #[limits_middle_part[1][0],limits_middle_part[1][1]]=[(limits_middle_part[1][0] + limits_middle_part[1][1]) / 2 + (limits_middle_part[1][0] - limits_middle_part[1][1]) * 0.1,(limits_middle_part[1][0] + limits_middle_part[1][1]) / 2 - (limits_middle_part[1][0] - limits_middle_part[1][1]) * 0.1]
        [limits_middle_part[1][0],limits_middle_part[1][1]]=[ (limits_middle_part[1][0] - limits_middle_part[1][1]) * 0.1, - (limits_middle_part[1][0] - limits_middle_part[1][1]) * 0.1]+(limits_middle_part[1][0] + limits_middle_part[1][1]) / 2
        '''
        #limits_middle_part[0][1] = 0.8 * (limits_middle_part[0][1] - limits_middle_part[0][0]) + limits_middle_part[0][0]
        #[limits_middle_part[0][0],limits_middle_part[0][1]]=[ (limits_middle_part[0][0] - limits_middle_part[0][1]) * 0.5, - (limits_middle_part[0][0] - limits_middle_part[0][1]) * 0.5]+(limits_middle_part[0][0] + limits_middle_part[0][1]) / 2

        #X_f_3 = LatinHypercubeSample(round(N_f), limits_middle_part)

        #range_list_2[0][0], range_list_2[0][1] = -1 * 0.5 + 1/2, - (0 - 1) * 0.5 + 1/2
        #range_list_2[0][0], range_list_2[0][1] = 0, - (0 - 1) * 0.5
        limits_middle = np.array(range_list_2)
        X_f_4 = LatinHypercubeSample(round(N_f*1), limits_middle)

        '''range_list_2[0][0], range_list_2[0][1] = -1 * 0.25 + 1/2, - (0 - 1) * 0.25 + 1/2
        limits_middle = np.array(range_list_2)
        X_f_3 = LatinHypercubeSample(round(N_f*0.25), limits_middle)'''

        #range_list[0][0], range_list[0][1] = 0, 1
        #N_f = 4.0##############

        #range_list[0][0], range_list[0][1] = -0.5, 0.5
        #n_2 = np.round(np.sqrt(N_f * (range_list[1][1] - range_list[1][0]) / (range_list[0][1] - range_list[0][0])))#/2
        n_2 = np.round(np.sqrt(N_f * points_scala))  # /2
        n_1 = np.round(N_f / n_2)
        R = np.linspace(range_list[0][0], range_list[0][1], n_1.astype(np.int32))############
        #R = np.linspace(range_list[0][0] + (range_list[0][1]-range_list[0][0])/n_1.astype(np.int32)/2,
        #                range_list[0][1] - (range_list[0][1]-range_list[0][0])/n_1.astype(np.int32)/2, n_1.astype(np.int32))  ############
        theta = np.linspace(range_list[1][0], range_list[1][1], n_2.astype(np.int32))############
        R, THETA = np.meshgrid(R, theta)
        X_f_2 = np.hstack((R.flatten()[:, None], THETA.flatten()[:, None]))

        n_2 = np.round(np.sqrt(N_f * points_scala))  # /2
        n_1 = np.round(N_f / n_2)
        #R = np.array([0.92-1e-3,0.92,0.92+1e-3])
        R = np.array([0.92])
        theta = np.linspace(range_list[1][0], range_list[1][1], n_2.astype(np.int32))############
        R, THETA = np.meshgrid(R, theta)
        X_f_3 = np.hstack((R.flatten()[:, None], THETA.flatten()[:, None]))

        n_2 = np.round(np.sqrt(N_f * points_scala))  # /2
        n_1 = np.round(N_f / n_2)
        R = np.linspace(range_list[0][0], 0.918, n_1.astype(np.int32))############
        theta = np.linspace(range_list[1][0], range_list[1][1], n_2.astype(np.int32))############
        R, THETA = np.meshgrid(R, theta)
        X_f_5_1 = np.hstack((R.flatten()[:, None], THETA.flatten()[:, None]))

        n_2 = np.round(np.sqrt(N_f * points_scala))  # /2
        n_1 = np.round(N_f / n_2)
        R = np.linspace(0.922, range_list[0][1], n_1.astype(np.int32))############
        theta = np.linspace(range_list[1][0], range_list[1][1], n_2.astype(np.int32))############
        R, THETA = np.meshgrid(R, theta)
        X_f_5_2 = np.hstack((R.flatten()[:, None], THETA.flatten()[:, None]))


        # self.X_f = X_f
        #X_f = np.concatenate((X_f_1, X_f_2), axis=0)
        #X_f = X_f_1
        X_f = X_f_2
        #X_f = X_f_3
        #X_f = np.concatenate((X_f_2, X_f_3), axis=0)  # 补充
        #X_f = np.concatenate((X_f_4, X_f_3), axis=0)  # 补充
        #X_f = np.concatenate((X_f_5_1, X_f_5_2), axis=0)  # 补充
        #self.X_f = X_f
        return X_f


    def add(self, token, vals, fidel):
        self.domain_ids.append(token)
        self.domaindict.append({
            "identifier": token,
            "range": vals,
            (token + "fidelity"): fidel,
            (token + "linspace"): np.linspace(vals[0], vals[1], fidel),
            (token + "upper"): vals[1],
            (token + "lower"): vals[0]
        })
