import numpy as np
import random
import copy
import matplotlib.pyplot as plt

#============== IMPORTING MODEL ======================================================================================

from Game_Net_Crow_Reasoning import *

input_dim      = 35
output_dim     = 25
alpha          = 0.1
epochs         = 1
beta           = 0.1
rounds         = 10000
Machine        = Game_Net_Problem_Solving(input_dim, output_dim, alpha, epochs, beta, rounds)

#============== TRAINING MODEL =======================================================================================

Machine.DFNN_to_RNN(input_dim - output_dim, output_dim)
Machine.synapse_layer_3_to_future_layer_1  = np.genfromtxt('self.synapse_layer_3_to_future_layer_1.csv', delimiter=',')
Machine.conduit_layer_3_to_future_layer_3  = np.genfromtxt('self.conduit_layer_3_to_future_layer_3.csv', delimiter=',')
Machine.synapse_layer_0_to_layer_1         = np.genfromtxt('self.synapse_layer_0_to_layer_1.csv', delimiter=',')
Machine.synapse_layer_1_to_layer_2         = np.genfromtxt('self.synapse_layer_1_to_layer_2.csv', delimiter=',')
Machine.synapse_layer_2_to_layer_3         = np.genfromtxt('self.synapse_layer_2_to_layer_3.csv', delimiter=',')


#============== NATRUAL FEEDBACK ====================================================================================


def examine(array):

    threshold = 0.5

    if (array[0] > threshold)&(array[1] > threshold) & ((np.argmax(array) == 0) | (np.argmax(array) == 1)):
        return "left"
    if (array[2] > threshold)&(array[3] > threshold) & ((np.argmax(array) == 2) | (np.argmax(array) == 3)):
        return "top"
    if (array[4] > threshold)&(array[5] > threshold) & ((np.argmax(array) == 4) | (np.argmax(array) == 5)):
        return "right"
    if (array[6] > threshold)&(array[7] > threshold) & ((np.argmax(array) == 6) | (np.argmax(array) == 7)):
        return "bottom"
    if (array[8] > threshold)&(array[9] > threshold) & ((np.argmax(array) == 8) | (np.argmax(array) == 9)):
        return "still"
    else:
        return "still"



def single_running(initial_map, array):

    spared_map = np.zeros_like(initial_map)
    row_index, column_index = np.where(initial_map == 1)


    if examine(array) == "left":
        if column_index[0] != 0:
            column_index[0] -= 1
            column_index[1] -= 1
            column_index[2] -= 1
            column_index[3] -= 1

    if examine(array) == "top":
        if row_index[0] != 0:
            row_index[0] -= 1
            row_index[1] -= 1
            row_index[2] -= 1
            row_index[3] -= 1

    if examine(array) == "right":
        if column_index[1] != 4:
            column_index[0] += 1
            column_index[1] += 1
            column_index[2] += 1
            column_index[3] += 1

    if examine(array) == "bottom":
        if row_index[2] != 4:
            row_index[0] += 1
            row_index[1] += 1
            row_index[2] += 1
            row_index[3] += 1
    if examine(array) == "still":
        row_index = row_index
        column_index = column_index


    spared_map[row_index[0]][column_index[0]] = 1
    spared_map[row_index[1]][column_index[1]] = 1
    spared_map[row_index[2]][column_index[2]] = 1
    spared_map[row_index[3]][column_index[3]] = 1
    return spared_map



#============== DEDUCTING USING MODELS ==============================================================================



random.seed()
start_map                      = np.array([[0, 0, 0, 0, 0],                                                               # <<<<<<<<<<<<<<<<<<
                                           [0, 0, 0, 0 ,0],
                                           [1, 1, 0, 0, 0],
                                           [1, 1, 0, 0, 0],
                                           [0, 0, 0, 0, 0]])

# start_map                      = np.array([[1, 1, 0, 0, 0],                                                               # <<<<<<<<<<<<<<<<<<
#                                            [1, 1, 0, 0 ,0],
#                                            [0, 0, 0, 0, 0],
#                                            [0, 0, 0, 0, 0],
#                                            [0, 0, 0, 0, 0]])
#
# start_map                      = np.array([[0, 0, 0, 0, 0],                                                               # <<<<<<<<<<<<<<<<<<
#                                            [0, 0, 0, 0 ,0],
#                                            [0, 0, 0, 0, 0],
#                                            [1, 1, 0, 0, 0],
#                                            [1, 1, 0, 0, 0]])
#
# start_map                      = np.array([[0, 0, 0, 0, 0],                                                               # <<<<<<<<<<<<<<<<<<
#                                            [0, 0, 0, 0 ,0],
#                                            [0, 0, 0, 0, 0],
#                                            [0, 0, 0, 1, 1],
#                                            [0, 0, 0, 1, 1]])
#

end_map                        = np.array([[0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0],
                                           [0, 0, 0, 1, 1],
                                           [0, 0, 0, 1, 1],
                                           [0, 0, 0, 0, 0]])




steps_remain                   = 4



for j in range(steps_remain):

    player_movements        = list()
    for i in range(steps_remain - j):
        player_movements.append((np.random.random(input_dim - output_dim) - 0.5) * 0.1)
    player_movements               = copy.deepcopy(np.array(player_movements))




    player_movements = Machine.deduct_from(start_map.flatten(), player_movements, end_map.flatten() )



    print(Machine.sigmoid(player_movements))


    print("The next movement is:")
    index = 0
    selected = Machine.sigmoid(player_movements[index])
    # while (examine(selected) == "still") & (index <= steps_remain - j - 2):
    #     index += 1
    #     selected = Machine.sigmoid(player_movements[index])
    print(examine(selected))
    print("====================================")

    start_map = single_running(start_map, selected)

    if (start_map == end_map).all():
        print("Congratulation!!!")
        break



