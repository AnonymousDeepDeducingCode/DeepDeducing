import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
import sys

np.random.seed(0)

initial_epochs   = 0                       #------------------------------------------------------------------------------------------------------------------------------------------ 20000

interval         = 200                        #------------------------------------------------------------------------------------------------------------------------------------------

times_epochs     = 150               #------------------------------------------------------------------------------------------------------------------------------------------

optimal_trials   = 50                           #------------------------------------------------------------------------------------------------------------------------------------------

size_of_pain_neurons  = 10

from Basic_Research_Model_for_test_2_particle_swarm_7_particles import *

dims                  = np.array([0, 15, 0])                                                                           #--------------------------- 12


alpha                 = 0.1                                                                                                 #---------------------------  0.05
initial_epochs        = initial_epochs                                                                                      #---------------------------

beta                  = 0.1                                                                                        #---------------------------
rounds                = 100                                                                                         #--------------------------- 20000
rounds_inside_swarm   = 100

signal_receiver       = "sigmoid"                                                                                   #----------------------------------------------------------  monotonic

signal_transfer       = "sigmoid"                                                                              #----------------------------------------------------------

function              = "sigmoid"                                                                                            #---------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------- 4X4 ------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

total_score_list    = np.zeros(times_epochs)


def finding_Nash(strategy_array, output_matrix):
    strategy_array = np.array(copy.deepcopy(strategy_array))
    output_matrix = np.array(copy.deepcopy(output_matrix))
    output_matrix = output_matrix.reshape((4, 4, 2))
    score = 0

    if (strategy_array[0] == 0) & (strategy_array[1] == 0):
        if \
                        (output_matrix[0][0][0] > output_matrix[0][1][0]) & \
                        (output_matrix[0][0][0] > output_matrix[0][2][0]) & \
                        (output_matrix[0][0][0] > output_matrix[0][3][0]) & \
                        (output_matrix[0][0][1] > output_matrix[1][0][1]) & \
                        (output_matrix[0][0][1] > output_matrix[2][0][1]) & \
                        (output_matrix[0][0][1] > output_matrix[3][0][1]):
            score = 1

    if (strategy_array[0] == 0) & (strategy_array[1] == 1):
        if \
                        (output_matrix[1][0][0] > output_matrix[1][1][0]) & \
                        (output_matrix[1][0][0] > output_matrix[1][2][0]) & \
                        (output_matrix[1][0][0] > output_matrix[1][3][0]) & \
                        (output_matrix[1][0][1] > output_matrix[0][0][1]) & \
                        (output_matrix[1][0][1] > output_matrix[2][0][1]) & \
                        (output_matrix[1][0][1] > output_matrix[3][0][1]):
            score = 1

    if (strategy_array[0] == 0) & (strategy_array[1] == 2):
        if \
                        (output_matrix[2][0][0] > output_matrix[2][1][0]) & \
                        (output_matrix[2][0][0] > output_matrix[2][2][0]) & \
                        (output_matrix[2][0][0] > output_matrix[2][3][0]) & \
                        (output_matrix[2][0][1] > output_matrix[1][0][1]) & \
                        (output_matrix[2][0][1] > output_matrix[0][0][1]) & \
                        (output_matrix[2][0][1] > output_matrix[3][0][1]):
            score = 1

    if (strategy_array[0] == 0) & (strategy_array[1] == 3):
        if \
                        (output_matrix[3][0][0] > output_matrix[3][1][0]) & \
                        (output_matrix[3][0][0] > output_matrix[3][2][0]) & \
                        (output_matrix[3][0][0] > output_matrix[3][3][0]) & \
                        (output_matrix[3][0][1] > output_matrix[1][0][1]) & \
                        (output_matrix[3][0][1] > output_matrix[2][0][1]) & \
                        (output_matrix[3][0][1] > output_matrix[0][0][1]):
            score = 1

    if (strategy_array[0] == 1) & (strategy_array[1] == 0):
        if \
                        (output_matrix[0][1][0] > output_matrix[0][0][0]) & \
                        (output_matrix[0][1][0] > output_matrix[0][2][0]) & \
                        (output_matrix[0][1][0] > output_matrix[0][3][0]) & \
                        (output_matrix[0][1][1] > output_matrix[1][1][1]) & \
                        (output_matrix[0][1][1] > output_matrix[2][1][1]) & \
                        (output_matrix[0][1][1] > output_matrix[3][1][1]):
            score = 1

    if (strategy_array[0] == 1) & (strategy_array[1] == 1):
        if \
                        (output_matrix[1][1][0] > output_matrix[1][0][0]) & \
                        (output_matrix[1][1][0] > output_matrix[1][2][0]) & \
                        (output_matrix[1][1][0] > output_matrix[1][3][0]) & \
                        (output_matrix[1][1][1] > output_matrix[0][1][1]) & \
                        (output_matrix[1][1][1] > output_matrix[2][1][1]) & \
                        (output_matrix[1][1][1] > output_matrix[3][1][1]):
            score = 1

    if (strategy_array[0] == 1) & (strategy_array[1] == 2):
        if \
                        (output_matrix[2][1][0] > output_matrix[2][0][0]) & \
                        (output_matrix[2][1][0] > output_matrix[2][2][0]) & \
                        (output_matrix[2][1][0] > output_matrix[2][3][0]) & \
                        (output_matrix[2][1][1] > output_matrix[1][1][1]) & \
                        (output_matrix[2][1][1] > output_matrix[0][1][1]) & \
                        (output_matrix[2][1][1] > output_matrix[3][1][1]):
            score = 1

    if (strategy_array[0] == 1) & (strategy_array[1] == 3):
        if \
                        (output_matrix[3][1][0] > output_matrix[3][0][0]) & \
                        (output_matrix[3][1][0] > output_matrix[3][2][0]) & \
                        (output_matrix[3][1][0] > output_matrix[3][3][0]) & \
                        (output_matrix[3][1][1] > output_matrix[1][1][1]) & \
                        (output_matrix[3][1][1] > output_matrix[2][1][1]) & \
                        (output_matrix[3][1][1] > output_matrix[0][1][1]):
            score = 1

    if (strategy_array[0] == 2) & (strategy_array[1] == 0):
        if \
                        (output_matrix[0][2][0] > output_matrix[0][1][0]) & \
                        (output_matrix[0][2][0] > output_matrix[0][0][0]) & \
                        (output_matrix[0][2][0] > output_matrix[0][3][0]) & \
                        (output_matrix[0][2][1] > output_matrix[1][2][1]) & \
                        (output_matrix[0][2][1] > output_matrix[2][2][1]) & \
                        (output_matrix[0][2][1] > output_matrix[3][2][1]):
            score = 1

    if (strategy_array[0] == 2) & (strategy_array[1] == 1):
        if \
                        (output_matrix[1][2][0] > output_matrix[1][1][0]) & \
                        (output_matrix[1][2][0] > output_matrix[1][0][0]) & \
                        (output_matrix[1][2][0] > output_matrix[1][3][0]) & \
                        (output_matrix[1][2][1] > output_matrix[0][2][1]) & \
                        (output_matrix[1][2][1] > output_matrix[2][2][1]) & \
                        (output_matrix[1][2][1] > output_matrix[3][2][1]):
            score = 1

    if (strategy_array[0] == 2) & (strategy_array[1] == 2):
        if \
                        (output_matrix[2][2][0] > output_matrix[2][1][0]) & \
                        (output_matrix[2][2][0] > output_matrix[2][0][0]) & \
                        (output_matrix[2][2][0] > output_matrix[2][3][0]) & \
                        (output_matrix[2][2][1] > output_matrix[1][2][1]) & \
                        (output_matrix[2][2][1] > output_matrix[0][2][1]) & \
                        (output_matrix[2][2][1] > output_matrix[3][2][1]):
            score = 1

    if (strategy_array[0] == 2) & (strategy_array[1] == 3):
        if \
                        (output_matrix[3][2][0] > output_matrix[3][1][0]) & \
                        (output_matrix[3][2][0] > output_matrix[3][0][0]) & \
                        (output_matrix[3][2][0] > output_matrix[3][3][0]) & \
                        (output_matrix[3][2][1] > output_matrix[1][2][1]) & \
                        (output_matrix[3][2][1] > output_matrix[2][2][1]) & \
                        (output_matrix[3][2][1] > output_matrix[0][2][1]):
            score = 1

    if (strategy_array[0] == 3) & (strategy_array[1] == 0):
        if \
                        (output_matrix[0][3][0] > output_matrix[0][1][0]) & \
                        (output_matrix[0][3][0] > output_matrix[0][2][0]) & \
                        (output_matrix[0][3][0] > output_matrix[0][0][0]) & \
                        (output_matrix[0][3][1] > output_matrix[1][3][1]) & \
                        (output_matrix[0][3][1] > output_matrix[2][3][1]) & \
                        (output_matrix[0][3][1] > output_matrix[3][3][1]):
            score = 1

    if (strategy_array[0] == 3) & (strategy_array[1] == 1):
        if \
                        (output_matrix[1][3][0] > output_matrix[1][1][0]) & \
                        (output_matrix[1][3][0] > output_matrix[1][2][0]) & \
                        (output_matrix[1][3][0] > output_matrix[1][0][0]) & \
                        (output_matrix[1][3][1] > output_matrix[0][3][1]) & \
                        (output_matrix[1][3][1] > output_matrix[2][3][1]) & \
                        (output_matrix[1][3][1] > output_matrix[3][3][1]):
            score = 1

    if (strategy_array[0] == 3) & (strategy_array[1] == 2):
        if \
                        (output_matrix[2][3][0] > output_matrix[2][1][0]) & \
                        (output_matrix[2][3][0] > output_matrix[2][2][0]) & \
                        (output_matrix[2][3][0] > output_matrix[2][0][0]) & \
                        (output_matrix[2][3][1] > output_matrix[1][3][1]) & \
                        (output_matrix[2][3][1] > output_matrix[0][3][1]) & \
                        (output_matrix[2][3][1] > output_matrix[3][3][1]):
            score = 1

    if (strategy_array[0] == 3) & (strategy_array[1] == 3):
        if \
                        (output_matrix[3][3][0] > output_matrix[3][1][0]) & \
                        (output_matrix[3][3][0] > output_matrix[3][2][0]) & \
                        (output_matrix[3][3][0] > output_matrix[3][0][0]) & \
                        (output_matrix[3][3][1] > output_matrix[1][3][1]) & \
                        (output_matrix[3][3][1] > output_matrix[2][3][1]) & \
                        (output_matrix[3][3][1] > output_matrix[0][3][1]):
            score = 1

    return score

def checking_Nash(output_matrix):
    output_matrix = np.array(copy.deepcopy(output_matrix))
    output_matrix = output_matrix.reshape((4, 4, 2))
    score = 0
    Nash  = 0
    strategy_array = np.zeros(2)
    for i in range(4):
        for j in range(4):
            strategy_array[0] = i
            strategy_array[1] = j

            if (strategy_array[0] == 0) & (strategy_array[1] == 0):
                if \
                                (output_matrix[0][0][0] > output_matrix[0][1][0]) & \
                                (output_matrix[0][0][0] > output_matrix[0][2][0]) & \
                                (output_matrix[0][0][0] > output_matrix[0][3][0]) & \
                                (output_matrix[0][0][1] > output_matrix[1][0][1]) & \
                                (output_matrix[0][0][1] > output_matrix[2][0][1]) & \
                                (output_matrix[0][0][1] > output_matrix[3][0][1]):
                    score += 1

            if (strategy_array[0] == 0) & (strategy_array[1] == 1):
                if \
                                (output_matrix[1][0][0] > output_matrix[1][1][0]) & \
                                (output_matrix[1][0][0] > output_matrix[1][2][0]) & \
                                (output_matrix[1][0][0] > output_matrix[1][3][0]) & \
                                (output_matrix[1][0][1] > output_matrix[0][0][1]) & \
                                (output_matrix[1][0][1] > output_matrix[2][0][1]) & \
                                (output_matrix[1][0][1] > output_matrix[3][0][1]):
                    score += 1

            if (strategy_array[0] == 0) & (strategy_array[1] == 2):
                if \
                                (output_matrix[2][0][0] > output_matrix[2][1][0]) & \
                                (output_matrix[2][0][0] > output_matrix[2][2][0]) & \
                                (output_matrix[2][0][0] > output_matrix[2][3][0]) & \
                                (output_matrix[2][0][1] > output_matrix[1][0][1]) & \
                                (output_matrix[2][0][1] > output_matrix[0][0][1]) & \
                                (output_matrix[2][0][1] > output_matrix[3][0][1]):
                    score += 1

            if (strategy_array[0] == 0) & (strategy_array[1] == 3):
                if \
                                (output_matrix[3][0][0] > output_matrix[3][1][0]) & \
                                (output_matrix[3][0][0] > output_matrix[3][2][0]) & \
                                (output_matrix[3][0][0] > output_matrix[3][3][0]) & \
                                (output_matrix[3][0][1] > output_matrix[1][0][1]) & \
                                (output_matrix[3][0][1] > output_matrix[2][0][1]) & \
                                (output_matrix[3][0][1] > output_matrix[0][0][1]):
                    score += 1

            if (strategy_array[0] == 1) & (strategy_array[1] == 0):
                if \
                                (output_matrix[0][1][0] > output_matrix[0][0][0]) & \
                                (output_matrix[0][1][0] > output_matrix[0][2][0]) & \
                                (output_matrix[0][1][0] > output_matrix[0][3][0]) & \
                                (output_matrix[0][1][1] > output_matrix[1][1][1]) & \
                                (output_matrix[0][1][1] > output_matrix[2][1][1]) & \
                                (output_matrix[0][1][1] > output_matrix[3][1][1]):
                    score += 1

            if (strategy_array[0] == 1) & (strategy_array[1] == 1):
                if \
                                (output_matrix[1][1][0] > output_matrix[1][0][0]) & \
                                (output_matrix[1][1][0] > output_matrix[1][2][0]) & \
                                (output_matrix[1][1][0] > output_matrix[1][3][0]) & \
                                (output_matrix[1][1][1] > output_matrix[0][1][1]) & \
                                (output_matrix[1][1][1] > output_matrix[2][1][1]) & \
                                (output_matrix[1][1][1] > output_matrix[3][1][1]):
                    score += 1

            if (strategy_array[0] == 1) & (strategy_array[1] == 2):
                if \
                                (output_matrix[2][1][0] > output_matrix[2][0][0]) & \
                                (output_matrix[2][1][0] > output_matrix[2][2][0]) & \
                                (output_matrix[2][1][0] > output_matrix[2][3][0]) & \
                                (output_matrix[2][1][1] > output_matrix[1][1][1]) & \
                                (output_matrix[2][1][1] > output_matrix[0][1][1]) & \
                                (output_matrix[2][1][1] > output_matrix[3][1][1]):
                    score += 1

            if (strategy_array[0] == 1) & (strategy_array[1] == 3):
                if \
                                (output_matrix[3][1][0] > output_matrix[3][0][0]) & \
                                (output_matrix[3][1][0] > output_matrix[3][2][0]) & \
                                (output_matrix[3][1][0] > output_matrix[3][3][0]) & \
                                (output_matrix[3][1][1] > output_matrix[1][1][1]) & \
                                (output_matrix[3][1][1] > output_matrix[2][1][1]) & \
                                (output_matrix[3][1][1] > output_matrix[0][1][1]):
                    score += 1

            if (strategy_array[0] == 2) & (strategy_array[1] == 0):
                if \
                                (output_matrix[0][2][0] > output_matrix[0][1][0]) & \
                                (output_matrix[0][2][0] > output_matrix[0][0][0]) & \
                                (output_matrix[0][2][0] > output_matrix[0][3][0]) & \
                                (output_matrix[0][2][1] > output_matrix[1][2][1]) & \
                                (output_matrix[0][2][1] > output_matrix[2][2][1]) & \
                                (output_matrix[0][2][1] > output_matrix[3][2][1]):
                    score += 1

            if (strategy_array[0] == 2) & (strategy_array[1] == 1):
                if \
                                (output_matrix[1][2][0] > output_matrix[1][1][0]) & \
                                (output_matrix[1][2][0] > output_matrix[1][0][0]) & \
                                (output_matrix[1][2][0] > output_matrix[1][3][0]) & \
                                (output_matrix[1][2][1] > output_matrix[0][2][1]) & \
                                (output_matrix[1][2][1] > output_matrix[2][2][1]) & \
                                (output_matrix[1][2][1] > output_matrix[3][2][1]):
                    score += 1

            if (strategy_array[0] == 2) & (strategy_array[1] == 2):
                if \
                                (output_matrix[2][2][0] > output_matrix[2][1][0]) & \
                                (output_matrix[2][2][0] > output_matrix[2][0][0]) & \
                                (output_matrix[2][2][0] > output_matrix[2][3][0]) & \
                                (output_matrix[2][2][1] > output_matrix[1][2][1]) & \
                                (output_matrix[2][2][1] > output_matrix[0][2][1]) & \
                                (output_matrix[2][2][1] > output_matrix[3][2][1]):
                    score += 1

            if (strategy_array[0] == 2) & (strategy_array[1] == 3):
                if \
                                (output_matrix[3][2][0] > output_matrix[3][1][0]) & \
                                (output_matrix[3][2][0] > output_matrix[3][0][0]) & \
                                (output_matrix[3][2][0] > output_matrix[3][3][0]) & \
                                (output_matrix[3][2][1] > output_matrix[1][2][1]) & \
                                (output_matrix[3][2][1] > output_matrix[2][2][1]) & \
                                (output_matrix[3][2][1] > output_matrix[0][2][1]):
                    score += 1

            if (strategy_array[0] == 3) & (strategy_array[1] == 0):
                if \
                                (output_matrix[0][3][0] > output_matrix[0][1][0]) & \
                                (output_matrix[0][3][0] > output_matrix[0][2][0]) & \
                                (output_matrix[0][3][0] > output_matrix[0][0][0]) & \
                                (output_matrix[0][3][1] > output_matrix[1][3][1]) & \
                                (output_matrix[0][3][1] > output_matrix[2][3][1]) & \
                                (output_matrix[0][3][1] > output_matrix[3][3][1]):
                    score += 1

            if (strategy_array[0] == 3) & (strategy_array[1] == 1):
                if \
                                (output_matrix[1][3][0] > output_matrix[1][1][0]) & \
                                (output_matrix[1][3][0] > output_matrix[1][2][0]) & \
                                (output_matrix[1][3][0] > output_matrix[1][0][0]) & \
                                (output_matrix[1][3][1] > output_matrix[0][3][1]) & \
                                (output_matrix[1][3][1] > output_matrix[2][3][1]) & \
                                (output_matrix[1][3][1] > output_matrix[3][3][1]):
                    score += 1

            if (strategy_array[0] == 3) & (strategy_array[1] == 2):
                if \
                                (output_matrix[2][3][0] > output_matrix[2][1][0]) & \
                                (output_matrix[2][3][0] > output_matrix[2][2][0]) & \
                                (output_matrix[2][3][0] > output_matrix[2][0][0]) & \
                                (output_matrix[2][3][1] > output_matrix[1][3][1]) & \
                                (output_matrix[2][3][1] > output_matrix[0][3][1]) & \
                                (output_matrix[2][3][1] > output_matrix[3][3][1]):
                    score += 1

            if (strategy_array[0] == 3) & (strategy_array[1] == 3):
                if \
                                (output_matrix[3][3][0] > output_matrix[3][1][0]) & \
                                (output_matrix[3][3][0] > output_matrix[3][2][0]) & \
                                (output_matrix[3][3][0] > output_matrix[3][0][0]) & \
                                (output_matrix[3][3][1] > output_matrix[1][3][1]) & \
                                (output_matrix[3][3][1] > output_matrix[2][3][1]) & \
                                (output_matrix[3][3][1] > output_matrix[0][3][1]):
                    score += 1

    if score == 1:
        Nash = 1
    return Nash

def encode(int, size):
    array= np.zeros(size)
    for i in range(int):

        array[i] = 1

    return array

def return_strategy(array):
    dummy = np.zeros_like(array)
    for i in range(array.shape[0]):
        if array[i] > 0.5:                                                              # -------------------------------- 0.5
            dummy[i] = 1
    array = dummy
    strategy  = 0
    for i in range(array.shape[0]):
        strategy += array[i] * (np.power(2 ,i))
    return strategy

print(' ')
print("progress of 4x4")
for i in range(optimal_trials):


    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 2+2*i))
    sys.stdout.flush()
    sleep(0.25)

    input_list = []
    input_0 = np.array([0, 0, 0, 0])
    input_list.append(input_0)
    input_0 = np.array([1, 0, 0, 0])
    input_list.append(input_0)
    input_0 = np.array([0, 1, 0, 0])
    input_list.append(input_0)
    input_0 = np.array([1, 1, 0, 0])
    input_list.append(input_0)

    input_0 = np.array([0, 0, 1, 0])
    input_list.append(input_0)
    input_0 = np.array([1, 0, 1, 0])
    input_list.append(input_0)
    input_0 = np.array([0, 1, 1, 0])
    input_list.append(input_0)
    input_0 = np.array([1, 1, 1, 0])
    input_list.append(input_0)
    input_0 = np.array([0, 0, 0, 1])
    input_list.append(input_0)
    input_0 = np.array([1, 0, 0, 1])
    input_list.append(input_0)
    input_0 = np.array([0, 1, 0, 1])
    input_list.append(input_0)
    input_0 = np.array([1, 1, 0, 1])
    input_list.append(input_0)
    input_0 = np.array([0, 0, 1, 1])
    input_list.append(input_0)
    input_0 = np.array([1, 0, 1, 1])
    input_list.append(input_0)
    input_0 = np.array([0, 1, 1, 1])
    input_list.append(input_0)
    input_0 = np.array([1, 1, 1, 1])
    input_list.append(input_0)
    input_list = np.array(copy.deepcopy(input_list))



    strategy_list = []
    output_list = []
    for dummy in range(4 * 4):
        index_1 = np.random.randint(size_of_pain_neurons + 1)
        index_2 = np.random.randint(size_of_pain_neurons + 1)
        output_0 = np.array([index_1, index_2])
        strategy_0 = np.concatenate((encode(index_1, size_of_pain_neurons), encode(index_2, size_of_pain_neurons)))
        output_list.append(output_0)
        strategy_list.append(strategy_0)
    output_list = np.array(output_list)
    strategy_list = np.array(strategy_list)

    while checking_Nash(output_list) != 1:

        strategy_list = []
        output_list = []
        for dummy in range(4 * 4):
            index_1 = np.random.randint(size_of_pain_neurons + 1)                                                            # ----------------------- 6
            index_2 = np.random.randint(size_of_pain_neurons + 1)
            output_0 = np.array([index_1, index_2])
            strategy_0 = np.concatenate((encode(index_1, size_of_pain_neurons), encode(index_2, size_of_pain_neurons)))
            output_list.append(output_0)
            strategy_list.append(strategy_0)
        output_list = np.array(output_list)
        strategy_list = np.array(strategy_list)



    #------------------------------------------IMPORTING MODEL------------------------------------------------------

    dims[0]                       = input_list.shape[1]

    dims[-1]                      = strategy_list.shape[1]

    Machine                       = Basic_Research_Model(dims, alpha, initial_epochs, beta, rounds, rounds_inside_swarm, signal_receiver, signal_transfer, function)

    Machine.fit(input_list, strategy_list)

    for j in range(total_score_list.shape[0]):

        Machine.epochs        = interval

        Machine.fit(input_list, strategy_list)

        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        strategy_neurons_size             = np.true_divide(np.int(input_list.shape[1]), 2)
        strategy_neurons_size             = np.int(strategy_neurons_size)


        random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0                    #
        random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0                    #





        random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A,
                                                                                               random_input_inner_for_player_B,
                                                                                               )

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        index_1                           = return_strategy(Machine.processor(random_input_inner_for_player_A))
        index_2                           = return_strategy(Machine.processor(random_input_inner_for_player_B))
        total_score                       = finding_Nash(np.array([index_1, index_2]), output_list)
        total_score_list[j]              += total_score


print("------------------------")
print(total_score_list)
print("------------------------")


x_ = np.linspace(0, initial_epochs + interval * times_epochs)
random_correctness = np.true_divide(np.int(optimal_trials), np.power(2, strategy_neurons_size) * np.power(2, strategy_neurons_size))
random_correctness = np.int(random_correctness)
y_ = x_ * 0 + random_correctness
plt.plot(x_, y_, "r--")


y = total_score_list
x = np.zeros_like(y)
for i in range(x.shape[0]):
    x[i] = initial_epochs + i * interval + interval
plt.plot(x, y, color = 'blue')
plt.ylabel(r'$number \, of \, NEs \, found$', fontsize=8)
plt.xlabel(r'$epochs$', fontsize=8)
plt.title(r'$Deep \, Deducing \, Accuracy \, for \, 4x4 \, game \, table \, under \, swarm \, optimization$', fontsize = 8)
plt.ylim([0, optimal_trials])









plt.show()


