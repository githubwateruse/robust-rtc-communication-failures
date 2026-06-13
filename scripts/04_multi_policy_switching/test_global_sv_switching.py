# -*- coding: utf-8 -*-








# import some modules
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import time
import os
import math
import xlrd
import xlsxwriter
from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages
import gc



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



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Environmental programming module: SWMM simulation
class BasicEnv(object):
    def __init__(self, inp_file, state_vector):
        self.input_file = inp_file
        self.sv = state_vector
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
        if self.sv == "SV0":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV1":
            self.state = np.array([self.T3.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV2":
            self.state = np.array([self.T2.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV3":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T6.depth])
        if self.sv == "SV4":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth])





        if self.sv == "T2 only":
            self.state = np.array([self.T2.depth])
        if self.sv == "T3 only":
            self.state = np.array([self.T3.depth])
        if self.sv == "T4 only":
            self.state = np.array([self.T4.depth])
        if self.sv == "T6 only":
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
        if self.sv == "SV0":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV1":
            self.state = np.array([self.T3.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV2":
            self.state = np.array([self.T2.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV3":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T6.depth])
        if self.sv == "SV4":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth])





        if self.sv == "T2 only":
            self.state = np.array([self.T2.depth])
        if self.sv == "T3 only":
            self.state = np.array([self.T3.depth])
        if self.sv == "T4 only":
            self.state = np.array([self.T4.depth])
        if self.sv == "T6 only":
            self.state = np.array([self.T6.depth])



        Step_CSO_volume_T1 = self.T1.statistics['flooding_volume'] - self.T1_CSO
        Step_CSO_volume_T2 = self.T2.statistics['flooding_volume'] - self.T2_CSO
        Step_CSO_volume_T3 = self.T3.statistics['flooding_volume'] - self.T3_CSO
        Step_CSO_volume_T4 = self.T4.statistics['flooding_volume'] - self.T4_CSO
        Step_CSO_volume_T5 = self.T5.statistics['flooding_volume'] - self.T5_CSO
        Step_CSO_volume_T6 = self.T6.statistics['flooding_volume'] - self.T6_CSO


        Weighted_CSO_volume = self.T1_weight * Step_CSO_volume_T1 + self.T2_weight * Step_CSO_volume_T2  \
                      + self.T3_weight * Step_CSO_volume_T3 + self.T4_weight * Step_CSO_volume_T4  \
                      + self.T5_weight * Step_CSO_volume_T5 + self.T6_weight * Step_CSO_volume_T6  \
                      + self.CSO7_weight * (self.CSO7.statistics['flooding_volume'] - self.CSO7_CSO) + self.CSO8_weight * (self.CSO8.statistics['flooding_volume'] - self.CSO8_CSO) \
                      + self.CSO9_weight * (self.CSO9.statistics['flooding_volume'] - self.CSO9_CSO) + self.CSO10_weight * (self.CSO10.statistics['flooding_volume'] - self.CSO10_CSO)


        Total_CSO_volume = Step_CSO_volume_T1 + Step_CSO_volume_T2 + Step_CSO_volume_T3 + Step_CSO_volume_T4 + Step_CSO_volume_T5 + Step_CSO_volume_T6  \
                      + (self.CSO7.statistics['flooding_volume'] - self.CSO7_CSO) + (self.CSO8.statistics['flooding_volume'] - self.CSO8_CSO) \
                      + (self.CSO9.statistics['flooding_volume'] - self.CSO9_CSO) + (self.CSO10.statistics['flooding_volume'] - self.CSO10_CSO)





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


        return self.state, Weighted_CSO_volume, Total_CSO_volume, done_steps



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
        if self.sv == "SV0":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV1":
            self.state = np.array([self.T3.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV2":
            self.state = np.array([self.T2.depth, self.T4.depth, self.T6.depth])
        if self.sv == "SV3":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T6.depth])
        if self.sv == "SV4":
            self.state = np.array([self.T2.depth, self.T3.depth, self.T4.depth])





        if self.sv == "T2 only":
            self.state = np.array([self.T2.depth])
        if self.sv == "T3 only":
            self.state = np.array([self.T3.depth])
        if self.sv == "T4 only":
            self.state = np.array([self.T4.depth])
        if self.sv == "T6 only":
            self.state = np.array([self.T6.depth])
        return self.state, self.T

    def close(self):
        self.sim.report()
        self.sim.close()
        del self.sim
        gc.collect()





def get_reward_combined_strategy(params_SV2, params_SV4, params_T2, params_T3, params_T4, params_T6, General_Patterns):


    Obs_pattern_T2 = General_Patterns[0]
    Obs_pattern_T3 = General_Patterns[1]
    Obs_pattern_T4 = General_Patterns[2]
    Obs_pattern_T6 = General_Patterns[3]

    ep_r = 0
    swmm_file = str(REPO_ROOT / "SWMM_Astlingen" / "Astlingen_SWMM_ES.inp")

    Weighted_CSO_Volume_series = []
    Total_CSO_Volume_series = []

    storm_env = BasicEnv(swmm_file, "SV0")
    Storm_state, EP_LEN = storm_env.reset()


    for t in range (EP_LEN):
        if Obs_pattern_T2[t] == 0 and Obs_pattern_T4[t] == 0 and Obs_pattern_T6[t] == 0:
            input_state = np.array([Storm_state[0],  Storm_state[2],  Storm_state[3]])
            Orifice_setting = get_action(params_SV2, input_state)

        elif Obs_pattern_T6[t] == 0:
            input_state = np.array([Storm_state[3]])
            Orifice_setting = get_action(params_T6, input_state)

        elif Obs_pattern_T2[t] == 0 and Obs_pattern_T3[t] == 0 and Obs_pattern_T4[t] == 0:
            input_state = np.array([Storm_state[0],  Storm_state[1],  Storm_state[2]])
            Orifice_setting = get_action(params_SV4, input_state)

        elif Obs_pattern_T3[t] == 0:
            input_state = np.array([Storm_state[1]])
            Orifice_setting = get_action(params_T3, input_state)

        elif Obs_pattern_T4[t] == 0:
            input_state = np.array([Storm_state[2]])
            Orifice_setting = get_action(params_T4, input_state)

        elif Obs_pattern_T2[t] == 0:
            input_state = np.array([Storm_state[0]])
            Orifice_setting = get_action(params_T2, input_state)

        else:
            # all observation sensors are in failure
            # the orifice_settings of BC are applied
            Orifice_setting = [0.2366, 0.6508, 0.3523, 0.4303]

        New_Storm_state, weighted_CSO_volume, total_CSO_volume, done_steps = storm_env.step(Orifice_setting)

        Weighted_CSO_Volume_series.append(weighted_CSO_volume)
        Total_CSO_Volume_series.append(total_CSO_volume)

        done_bool = float(done_steps)


        Storm_state = New_Storm_state


        if done_bool:
            storm_env.close()
            break


    return Weighted_CSO_Volume_series, Total_CSO_Volume_series


def read_xlsx(base_path, sensor, duration, parallel_num):

    path = base_path + sensor + "_" + duration + "_" + str(parallel_num ) + ".xlsx"
    workbook = xlrd.open_workbook(path)
    data_sheet = workbook.sheet_by_index(0)
    rowNum = data_sheet.nrows
    colNum = data_sheet.ncols
    data_ = np.array(data_sheet.col_values(0))

    workbook.release_resources()
    del workbook
    return data_





def load_communication_pattern_General(duration, failure_probability, parallel_num):
    base_path = str(REPO_ROOT / "data_required" / "failure_patterns" / f"probability {failure_probability}") + "/"

    pattern_1 = read_xlsx(base_path, "T2", duration, parallel_num)
    pattern_2 = read_xlsx(base_path, "T3", duration, parallel_num)
    pattern_3 = read_xlsx(base_path, "T4", duration, parallel_num)
    pattern_4 = read_xlsx(base_path, "T6", duration, parallel_num)
    PATTERN = [pattern_1, pattern_2, pattern_3, pattern_4]
    return PATTERN



def Net_shapes_vs_params(state_vector, policy_index, len_state, len_action):
    policy_folder = state_vector.replace(" only", "")
    policy_path = str(REPO_ROOT / "trained_policies" / policy_folder / f"Gen_{policy_index}.xlsx")
    net_shapes, net_params = build_net(len_state, len_action, 0)


    workbook = xlrd.open_workbook(policy_path)
    data_sheet = workbook.sheet_by_index(0)
    rowNum = data_sheet.nrows
    colNum = data_sheet.ncols
    net_params = np.array(data_sheet.col_values(0))

    workbook.release_resources()
    del workbook

    network_params = params_reshape(net_shapes, net_params)

    return network_params


if __name__ == "__main__":
    Train_time_start = time.time()

    failure_probability = 0.01 # 0.01 or 0.001




    duration = "48h" # "12h", "24h" or "48h"

    for seed in range(10):
        parallel_num = seed


        General_Patterns = load_communication_pattern_General(duration, failure_probability, parallel_num)



        LEN_ACTION = 4

        policy_index_SV2 = 308
        policy_index_SV4 = 365

        policy_index_T2 = 290
        policy_index_T3 = 376
        policy_index_T4 = 399
        policy_index_T6 = 226


        print("parallel_num", parallel_num, "duration", duration)


        params_SV2 = Net_shapes_vs_params("SV2", policy_index_SV2, 3, 4)
        params_SV4 = Net_shapes_vs_params("SV4", policy_index_SV4, 3, 4)

        params_T2 = Net_shapes_vs_params("T2 only", policy_index_T2, 1, 4)
        params_T3 = Net_shapes_vs_params("T3 only", policy_index_T3, 1, 4)
        params_T4 = Net_shapes_vs_params("T4 only", policy_index_T4, 1, 4)
        params_T6 = Net_shapes_vs_params("T6 only", policy_index_T6, 1, 4)


        Weighted_CSO_Volume_series, Total_CSO_Volume_series = get_reward_combined_strategy(params_SV2, params_SV4, params_T2, params_T3, params_T4, params_T6, General_Patterns)


        logfile = open(REPO_ROOT / "SWMM_Astlingen" / "Astlingen_SWMM_ES.rpt", "r", encoding="utf-8")



        kws_start = ["Node Flooding Summary",]
        kws_end = ["Storage Volume Summary",]
        lines = logfile.readlines()
        # print(len(lines))
        for m in range (len(lines)):
            if (any (kw in lines[m] for kw in kws_start)):
                start_lines = m + 10
            if (any (kw in lines[m] for kw in kws_end)):
                end_lines = m - 4

        # print(lines[Report_start_lines])
        AAA = []
        BBB = []
        BBB_creek = []
        BBB_river = []
        for n in range (start_lines, end_lines+1):
            Res = lines[n].split('    ')
            # print(Res[0], np.round(float(Res[-2]), decimals = 3))
            AAA.append(Res[0])
            cso_volume = np.round(float(Res[-2]), decimals = 3)
            BBB.append(cso_volume)

            if Res[0] == '  CSO7' or Res[0] == '  CSO9' or Res[0] == '  T6':
                BBB_creek.append(cso_volume)
            else:
                BBB_river.append(cso_volume)
        print("total", np.sum(BBB))
        print("creek", np.sum(BBB_creek))
        print("river", np.sum(BBB_river))
        print("====================================================")


        logfile.close()


        output_path = REPO_ROOT / "data_required" / "simulation_outputs" / "global_sv_switching" / f"{duration}_pattern_{parallel_num}.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        workbook_curve = xlsxwriter.Workbook(str(output_path))
        worksheet_curve = workbook_curve.add_worksheet('sheet1')

        worksheet_curve.write_column(0,0, [np.sum(BBB)])
        worksheet_curve.write_column(0,1, [np.sum(BBB_creek)])
        worksheet_curve.write_column(0,2, [np.sum(BBB_river)])

        worksheet_curve.write_column(0,4, AAA)
        worksheet_curve.write_column(0,5, BBB)


        worksheet_curve.write_column(0,7, Weighted_CSO_Volume_series)
        worksheet_curve.write_column(0,8, Total_CSO_Volume_series)

        workbook_curve.close()

        del worksheet_curve
        del workbook_curve

        del AAA, BBB, BBB_creek, BBB_river, Weighted_CSO_Volume_series, Total_CSO_Volume_series


    Train_time_end = time.time()
    print("total time consumption", Train_time_end - Train_time_start)
