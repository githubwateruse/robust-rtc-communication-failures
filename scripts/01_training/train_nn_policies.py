# -*- coding: utf-8 -*-







# import some modules
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import multiprocessing as mp
import time
import os
import math
import xlrd
from xlrd import xldate_as_datetime, xldate_as_tuple
import xlsxwriter
from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages
import gc
from datetime import datetime

# Hyperparameters for Evolution Strategies
N_KID = 10

GENERATION_OFFSET = 0
N_GENERATION = 400



LR = .05                  # learning rate
SIGMA = .05                 # mutation strength or step size

State_Vector = "SV0"




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Environmental programming module: SWMM simulation
class BasicEnv(object):
    def __init__(self, inp_file):
        self.input_file = inp_file
        self.control_time_step = 300  # control time step in seconds

        # initialize Simulation
        self.sim = Simulation(self.input_file)  # read input file
        self.sim.step_advance(self.control_time_step)  # set control time step

        # initialize Objects
        self._init_objects()


        self.sim.start()
        self.t = 1

        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds()/self.control_time_step)


        # State vector formnulation
        if State_Vector == "SV0":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV1":
            self.state = np.array([self.T3.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV2":
            self.state = np.array([self.T2.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV3":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T6.depth])
        if State_Vector == "SV4":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth])





        if State_Vector == "T2 only":
            self.state = np.array([self.T2.depth])
        if State_Vector == "T3 only":
            self.state = np.array([self.T3.depth])
        if State_Vector == "T4 only":
            self.state = np.array([self.T4.depth])
        if State_Vector == "T6 only":
            self.state = np.array([self.T6.depth])

    def _init_objects(self):
        # init node object, link object, and subcatchment object
        subcatchment_object = Subcatchments(self.sim)
        self.SC01 = subcatchment_object["SC01"]
        self.SC02 = subcatchment_object["SC02"]
        self.SC03 = subcatchment_object["SC03"]
        self.SC04 = subcatchment_object["SC04"]
        self.SC05 = subcatchment_object["SC05"]
        self.SC06 = subcatchment_object["SC06"]
        self.SC07 = subcatchment_object["SC07"]
        self.SC08 = subcatchment_object["SC08"]
        self.SC09 = subcatchment_object["SC09"]
        self.SC010 = subcatchment_object["SC010"]

        node_object = Nodes(self.sim)
        self.T1 = node_object["T1"]; self.T2 = node_object["T2"]; self.T3 = node_object["T3"]
        self.T4 = node_object["T4"]; self.T5 = node_object["T5"]; self.T6 = node_object["T6"]
        self.CSO7 = node_object["CSO7"]; self.CSO8 = node_object["CSO8"]
        self.CSO9 = node_object["CSO9"]; self.CSO10 = node_object["CSO10"]

        self.J1 = node_object["J1"]
        self.J2 = node_object["J2"]
        self.J3 = node_object["J3"]
        self.J4 = node_object["J4"]
        self.J6 = node_object["J6"]
        self.J7 = node_object["J7"]
        self.J9 = node_object["J9"]
        self.J10 = node_object["J10"]
        self.J11 = node_object["J11"]
        self.J14 = node_object["J14"]
        self.J15 = node_object["J15"]
        self.J16 = node_object["J16"]
        self.J17 = node_object["J17"]
        self.J18 = node_object["J18"]
        self.J19 = node_object["J19"]
        self.Out_to_WWTP = node_object["Out_to_WWTP"]



        link_object = Links(self.sim)
        self.C14 = link_object["C14"]
        self.V1 = link_object["V1"]; self.V2 = link_object["V2"]; self.V3 = link_object["V3"]
        self.V4 = link_object["V4"]; self.V5 = link_object["V5"]; self.V6 = link_object["V6"]


        raingage_object = RainGages(self.sim)
        self.RG1 = raingage_object["RG1"]; self.RG2 = raingage_object["RG2"]
        self.RG3 = raingage_object["RG3"]; self.RG4 = raingage_object["RG4"]



        # the reward weights of the 6 storage units and 4 CSOs
        self.T1_weight, self.T2_weight, self.T3_weight, self.T4_weight, self.T5_weight, self.T6_weight = 1, 1, 1, 1, 1, 2
        self.CSO7_weight, self.CSO8_weight, self.CSO9_weight, self.CSO10_weight = 2, 1, 2, 1



        # the statistics of CSO volume of each node
        self.T1_CSO, self.T2_CSO, self.T3_CSO, self.T4_CSO, self.T5_CSO, self.T6_CSO = 0, 0, 0, 0, 0, 0
        self.CSO7_CSO, self.CSO8_CSO, self.CSO9_CSO, self.CSO10_CSO = 0, 0, 0, 0


    def step(self, action):
        # four orifices
        self.V2.target_setting = np.round(np.double(action[0]), decimals = 2)
        self.V3.target_setting = np.round(np.double(action[1]), decimals = 2)
        self.V4.target_setting = np.round(np.double(action[2]), decimals = 2)
        self.V6.target_setting = np.round(np.double(action[3]), decimals = 2)


        self.sim.__next__()


        # State vector formnulation
        if State_Vector == "SV0":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV1":
            self.state = np.array([self.T3.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV2":
            self.state = np.array([self.T2.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV3":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T6.depth])
        if State_Vector == "SV4":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth])






        if State_Vector == "T2 only":
            self.state = np.array([self.T2.depth])
        if State_Vector == "T3 only":
            self.state = np.array([self.T3.depth])
        if State_Vector == "T4 only":
            self.state = np.array([self.T4.depth])
        if State_Vector == "T6 only":
            self.state = np.array([self.T6.depth])



        Step_CSO_volume_T1 = self.T1.statistics['flooding_volume'] - self.T1_CSO
        Step_CSO_volume_T2 = self.T2.statistics['flooding_volume'] - self.T2_CSO
        Step_CSO_volume_T3 = self.T3.statistics['flooding_volume'] - self.T3_CSO
        Step_CSO_volume_T4 = self.T4.statistics['flooding_volume'] - self.T4_CSO
        Step_CSO_volume_T5 = self.T5.statistics['flooding_volume'] - self.T5_CSO
        Step_CSO_volume_T6 = self.T6.statistics['flooding_volume'] - self.T6_CSO


        Step_CSO_volume = self.T1_weight * Step_CSO_volume_T1 + self.T2_weight * Step_CSO_volume_T2  \
                      + self.T3_weight * Step_CSO_volume_T3 + self.T4_weight * Step_CSO_volume_T4  \
                      + self.T5_weight * Step_CSO_volume_T5 + self.T6_weight * Step_CSO_volume_T6  \
                      + self.CSO7_weight * (self.CSO7.statistics['flooding_volume'] - self.CSO7_CSO) + self.CSO8_weight * (self.CSO8.statistics['flooding_volume'] - self.CSO8_CSO) \
                      + self.CSO9_weight * (self.CSO9.statistics['flooding_volume'] - self.CSO9_CSO) + self.CSO10_weight * (self.CSO10.statistics['flooding_volume'] - self.CSO10_CSO)


        Reward_CSO_volume = -1 * Step_CSO_volume/1000



        self.T1_CSO, self.T2_CSO = self.T1.statistics['flooding_volume'], self.T2.statistics['flooding_volume']
        self.T3_CSO, self.T4_CSO = self.T3.statistics['flooding_volume'], self.T4.statistics['flooding_volume']
        self.T5_CSO, self.T6_CSO = self.T5.statistics['flooding_volume'], self.T6.statistics['flooding_volume']

        self.CSO7_CSO, self.CSO8_CSO = self.CSO7.statistics['flooding_volume'], self.CSO8.statistics['flooding_volume']
        self.CSO9_CSO, self.CSO10_CSO = self.CSO9.statistics['flooding_volume'], self.CSO10.statistics['flooding_volume']
        # print(self.t, self.T4_CSO)

        if self.t < self.T-1:
            done_steps = False
        else:
            done_steps = True

        self.t += 1


        return self.state, Reward_CSO_volume, done_steps



    def reset(self):

        # self.sim.close()
        # del self.sim
        # gc.collect()

        # initialize Simulation
        self.sim = Simulation(self.input_file)  # read input file
        self.sim.step_advance(self.control_time_step)  # set control time step


        # initialize Objects
        self._init_objects()
        self.sim.start()
        self.t = 1

        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds()/self.control_time_step)
        # print("total control steps", self.T)


        # State vector formnulation
        if State_Vector == "SV0":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV1":
            self.state = np.array([self.T3.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV2":
            self.state = np.array([self.T2.depth, self.T4.depth, self.T6.depth])
        if State_Vector == "SV3":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T6.depth])
        if State_Vector == "SV4":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth])




        if State_Vector == "T2 only":
            self.state = np.array([self.T2.depth])
        if State_Vector == "T3 only":
            self.state = np.array([self.T3.depth])
        if State_Vector == "T4 only":
            self.state = np.array([self.T4.depth])
        if State_Vector == "T6 only":
            self.state = np.array([self.T6.depth])
        return self.state, self.T

    def close(self):
        self.sim.report()
        self.sim.close()
        del self.sim
        gc.collect()



# some functions of Evolution Strategies
# --------------------------------------------------------------------------------
def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params, dtype = np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v


def params_reshape(shapes, params):  # reshape to be a matrix
    p, start = [], 0
    for shape in shapes:
        n_w, n_b = shape[0] * shape[1], shape[1]
        p.append(params[start: start + n_w].reshape(shape))
        p.append(params[start + n_w: start + n_w + n_b].reshape((1, shape[1])))
        start += n_w + n_b
    return p




def build_net(len_state, len_action, random_seed):
    rng = np.random.default_rng(random_seed)
    def linear(n_in, n_out):  # network linear layer
        w = rng.standard_normal(n_in * n_out, dtype=np.float32) * 0.1
        b = rng.standard_normal(n_out, dtype=np.float32) * 0.1
        return (n_in, n_out), np.concatenate((w, b))

    s0, p0 = linear(len_state, 30)
    s1, p1 = linear(30, 30)
    s2, p2 = linear(30, len_action)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))



def get_action(params, x):
    x = x[np.newaxis, :]

    x = x.dot(params[0])
    x += params[1]
    np.maximum(x, 0, out=x)

    x = x.dot(params[2])
    x += params[3]
    np.maximum(x, 0, out=x)

    x = x.dot(params[4])
    x += params[5]
    return 0.5 * np.tanh(x)[0] + 0.5  # continuous action



def get_reward_per_event(network_params):
    ep_r = 0
    swmm_file = str(REPO_ROOT / "SWMM_Astlingen" / "Astlingen_SWMM_RL_synthetic_year_2000.inp")


    storm_env = BasicEnv(swmm_file)
    Storm_state, EP_LEN = storm_env.reset()

    for t in range (EP_LEN):
        Orifice_setting = get_action(network_params, Storm_state)
        New_Storm_state, R, done_steps = storm_env.step(Orifice_setting)


        done_bool = float(done_steps)

        ep_r = ep_r + R
        Storm_state = New_Storm_state


        if done_bool:
            storm_env.close()
            break
    return ep_r



def get_reward_training_events(shapes, params, seed_and_id=None):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id

        rng = np.random.default_rng(seed)
        params += sign(k_id) * SIGMA * rng.standard_normal(params.size, dtype=np.float32)


    network_params = params_reshape(shapes, params)
    EP_R = get_reward_per_event(network_params)

    return EP_R



# Train
# --------------------------------------------------------------------------------
def train(net_shapes, net_params, optimizer, utility, num_generation):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.arange(num_generation*N_KID + 1, num_generation*N_KID + N_KID + 1).repeat(2)  # mirrored sampling

    rewards = np.empty(2 * N_KID, dtype=np.float32)
    for k_id in range(N_KID*2):
        reward = get_reward_training_events(net_shapes, net_params, [noise_seed[k_id], k_id])
        rewards[k_id] = reward


    """ update with reward rank and utility """
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

    cumulative_update = np.zeros_like(net_params)       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        rng = np.random.default_rng(noise_seed[k_id]) # reconstruct noise using seed
        cumulative_update += utility[ui] * sign(k_id) * rng.standard_normal(net_params.size, dtype=np.float32)


    gradients = optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))
    return net_params + gradients





if __name__ == "__main__":
    Train_time_start = time.time()

    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)

    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base


    if State_Vector == "SV1" or State_Vector == "SV2" or State_Vector == "SV3" or State_Vector == "SV4":
        LEN_STATE = 3
    elif State_Vector == "T2 only" or State_Vector == "T3 only" or State_Vector == "T4 only" or State_Vector == "T6 only":
        LEN_STATE = 1
    else:
        LEN_STATE = 4
    LEN_ACTION = 4
    result_path = REPO_ROOT / "data_required" / "training_runs" / State_Vector.replace(" only", "")
    result_path.mkdir(parents=True, exist_ok=True)


    # training
    # the structure of each neural network is identical, so their shapes are the same
    net_shapes, net_params = build_net(LEN_STATE, LEN_ACTION, 0)
    optimizer = SGD(net_params, LR)

    if GENERATION_OFFSET != 0:
        path_0 = result_path / f"Gen_{GENERATION_OFFSET - 1}.xlsx"
        workbook = xlrd.open_workbook(str(path_0))
        data_sheet = workbook.sheet_by_index(0)
        rowNum = data_sheet.nrows
        colNum = data_sheet.ncols
        net_params = np.array(data_sheet.col_values(0))




    for g in range(N_GENERATION):
        t0 = time.time()

        num_generation = g + GENERATION_OFFSET
        net_params = train(net_shapes, net_params, optimizer, utility, num_generation)

        # test trained net without noise
        net_r = get_reward_training_events(net_shapes, net_params, seed_and_id=None)

        print(
            '| Gen: ',num_generation,
            '| Net_R: %.3f' % net_r,
            # '| Num_Spills_: %.1f' % num_spills,
            '| Gen_T: %.2f' % (time.time() - t0),)

        training_time = time.time() - Train_time_start
        # record the parameters in each generation
        workbook_params = xlsxwriter.Workbook(str(result_path / f"Gen_{num_generation}.xlsx"))
        worksheet_params = workbook_params.add_worksheet('sheet1')
        worksheet_params.write_column(0,0, net_params)

        worksheet_params.write_column(0,1, [net_r])
        worksheet_params.write_column(0,2,  [training_time])
        workbook_params.close()

    Train_time_end = time.time()
    print("total time consumption", Train_time_end - Train_time_start)
