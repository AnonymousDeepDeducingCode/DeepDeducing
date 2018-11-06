import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)



f, axarr = plt.subplots(2, 2,figsize=(12, 12))

number_of_tested_neurons = 50


print("test")


#--------------------------- MODEL 1--------------------------------------------------------------------------------

input_list = []
input_0 = np.array([0, 0])
input_list.append(input_0)
input_0 = np.array([1, 0])
input_list.append(input_0)
input_0 = np.array([0, 1])
input_list.append(input_0)
input_0 = np.array([1, 1])
input_list.append(input_0)

output_list = []
output_0 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
output_list.append(output_0)

input_list  = np.array(input_list)
output_list = np.array(output_list)

#------------------------------------------IMPORTING MODEL------------------------------------------------------

from Basic_Research_Model_for_test_1 import *

input_dim             = input_list.shape[1]
output_dim            = output_list.shape[1]

alpha                 = 0.01
epochs                = 10000                                                                                       #---------------------------

beta                  = 0.01
rounds                = 10000                                                                                      #---------------------------

pruning               = "not True"                                                                                      #---------------------------
pruning_criterion     = 0.225                                                                                           #---------------------------
stabilizing           = "not True"                                                                                         #---------------------------
stabilizing_criterion = 1                                                                                           #---------------------------

randomness            = "not True"                                                                                      #---------------------------

function              = "sigmoid"

Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds, pruning, pruning_criterion, stabilizing, stabilizing_criterion, randomness, function)

Machine.fit(input_list, output_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for i in range(number_of_tested_neurons):

    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = random_input_inner_for_player_A
    saved_random_input_inner_for_player_B = random_input_inner_for_player_B

    axarr[0][0].plot(np.round(Machine.processor(saved_random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(saved_random_input_inner_for_player_A), 2)[0], 'gs', zorder=9999999)

    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr[0][0])

    axarr[0][0].plot(np.round(Machine.processor(random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(random_input_inner_for_player_A), 2)[0], 'rs')


axarr[0][0].set_ylabel(r'$player \, A \, strategy$', fontsize=10)
axarr[0][0].set_xlim([0, 1])
axarr[0][0].set_ylim([1, 0])




#--------------------------- MODEL 2--------------------------------------------------------------------------------

input_list = []
input_0 = np.array([0, 0])
input_list.append(input_0)
input_0 = np.array([1, 0])
input_list.append(input_0)
input_0 = np.array([0, 1])
input_list.append(input_0)
input_0 = np.array([1, 1])
input_list.append(input_0)


output_list = []
output_0 = np.array([0.7, 0.7])
output_list.append(output_0)
output_0 = np.array([1.0, 0.0])
output_list.append(output_0)
output_0 = np.array([0.0, 1.0])
output_list.append(output_0)
output_0 = np.array([0.5, 0.5])
output_list.append(output_0)

output_list = []
output_0 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
output_list.append(output_0)


input_list  = np.array(input_list)
output_list = np.array(output_list)

#------------------------------------------IMPORTING MODEL------------------------------------------------------

from Basic_Research_Model_for_test_1 import *

input_dim             = input_list.shape[1]
output_dim            = output_list.shape[1]

alpha                 = 0.01
epochs                = 10000                                                                                       #---------------------------

beta                  = 0.01
rounds                = 10000                                                                                      #---------------------------

pruning               = "not True"                                                                                      #---------------------------
pruning_criterion     = 0.225                                                                                           #---------------------------
stabilizing           = "not True"                                                                                         #---------------------------
stabilizing_criterion = 1                                                                                           #---------------------------

randomness            = "not True"                                                                                      #---------------------------

function              = "sigmoid"

Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds, pruning, pruning_criterion, stabilizing, stabilizing_criterion, randomness, function)

Machine.fit(input_list, output_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for i in range(number_of_tested_neurons):

    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = random_input_inner_for_player_A
    saved_random_input_inner_for_player_B = random_input_inner_for_player_B

    axarr[0][1].plot(np.round(Machine.processor(saved_random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(saved_random_input_inner_for_player_A), 2)[0], 'gs', zorder =9999999)

    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr[0][1])

    axarr[0][1].plot(np.round(Machine.processor(random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(random_input_inner_for_player_A), 2)[0], 'rs')

axarr[0][1].set_xlim([0, 1])
axarr[0][1].set_ylim([1, 0])




#--------------------------- MODEL 3--------------------------------------------------------------------------------

input_list = []
input_0 = np.array([0, 0])
input_list.append(input_0)
input_0 = np.array([1, 0])
input_list.append(input_0)
input_0 = np.array([0, 1])
input_list.append(input_0)
input_0 = np.array([1, 1])
input_list.append(input_0)

output_list = []
output_0 = np.array([1.0, 0.0])
output_list.append(output_0)
output_0 = np.array([0.5, 0.5])
output_list.append(output_0)
output_0 = np.array([0.7, 0.7])
output_list.append(output_0)
output_0 = np.array([0.0, 1.0])
output_list.append(output_0)

output_list = []
output_0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
output_list.append(output_0)

input_list  = np.array(input_list)
output_list = np.array(output_list)

#------------------------------------------IMPORTING MODEL------------------------------------------------------

from Basic_Research_Model_for_test_1 import *

input_dim             = input_list.shape[1]
output_dim            = output_list.shape[1]

alpha                 = 0.01
epochs                = 10000                                                                                       #---------------------------

beta                  = 0.01
rounds                = 10000                                                                                      #---------------------------

pruning               = "not True"                                                                                      #---------------------------
pruning_criterion     = 0.225                                                                                           #---------------------------
stabilizing           = "not True"                                                                                         #---------------------------
stabilizing_criterion = 1                                                                                           #---------------------------

randomness            = "not True"                                                                                      #---------------------------

function              = "sigmoid"

Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds, pruning, pruning_criterion, stabilizing, stabilizing_criterion, randomness, function)

Machine.fit(input_list, output_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for i in range(number_of_tested_neurons):

    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = random_input_inner_for_player_A
    saved_random_input_inner_for_player_B = random_input_inner_for_player_B

    axarr[1][0].plot(np.round(Machine.processor(saved_random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(saved_random_input_inner_for_player_A), 2)[0], 'gs', zorder =9999999)

    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr[1][0])

    axarr[1][0].plot(np.round(Machine.processor(random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(random_input_inner_for_player_A), 2)[0], 'rs')


axarr[1][0].set_xlabel(r'$player \, B \, strategy$', fontsize=10)
axarr[1][0].set_ylabel(r'$player \, A \, strategy$', fontsize=10)
axarr[1][0].set_xlim([0, 1])
axarr[1][0].set_ylim([1, 0])




#--------------------------- MODEL 4--------------------------------------------------------------------------------

input_list = []
input_0 = np.array([0, 0])
input_list.append(input_0)
input_0 = np.array([1, 0])
input_list.append(input_0)
input_0 = np.array([0, 1])
input_list.append(input_0)
input_0 = np.array([1, 1])
input_list.append(input_0)

output_list = []
output_0 = np.array([0.0, 1.0])
output_list.append(output_0)
output_0 = np.array([0.7, 0.7])
output_list.append(output_0)
output_0 = np.array([0.5, 0.5])
output_list.append(output_0)
output_0 = np.array([1.0, 0.0])
output_list.append(output_0)

output_list = []
output_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
output_list.append(output_0)
output_0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
output_list.append(output_0)

input_list  = np.array(input_list)
output_list = np.array(output_list)

#------------------------------------------IMPORTING MODEL------------------------------------------------------

from Basic_Research_Model_for_test_1 import *

input_dim             = input_list.shape[1]
output_dim            = output_list.shape[1]

alpha                 = 0.01
epochs                = 10000                                                                                       #---------------------------

beta                  = 0.01
rounds                = 10000                                                                                      #---------------------------

pruning               = "not True"                                                                                      #---------------------------
pruning_criterion     = 0.225                                                                                           #---------------------------
stabilizing           = "not True"                                                                                         #---------------------------
stabilizing_criterion = 1                                                                                           #---------------------------

randomness            = "not True"                                                                                      #---------------------------

function              = "sigmoid"

Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds, pruning, pruning_criterion, stabilizing, stabilizing_criterion, randomness, function)

Machine.fit(input_list, output_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for i in range(number_of_tested_neurons):

    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = random_input_inner_for_player_A
    saved_random_input_inner_for_player_B = random_input_inner_for_player_B

    axarr[1][1].plot(np.round(Machine.processor(saved_random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(saved_random_input_inner_for_player_A), 2)[0], 'gs', zorder =9999999)

    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr[1][1])

    axarr[1][1].plot(np.round(Machine.processor(random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(random_input_inner_for_player_A), 2)[0], 'rs')

axarr[1][1].set_xlabel(r'$player \, B \, strategy$', fontsize=10)
axarr[1][1].set_xlim([0, 1])
axarr[1][1].set_ylim([1, 0])










plt.show()





