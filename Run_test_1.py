import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import sleep

import sys
np.random.seed(0)



f, axarr = plt.subplots(2, 2,figsize=(12, 12))

number_of_tested_neurons = 50


from Basic_Research_Model_for_test_1 import *

input_dim             = 1+1
output_dim            = 10+10

alpha                 = 0.01
epochs                = 10000                                                                                       #---------------------------

beta                  = 0.01
rounds                = 10000                                                                                      #---------------------------

function              = "sigmoid"


color_start           = 'red'
marker_start          = '.'
color_middle          =  ((238 ) / 255, (130) / 255, (238 ) / 255)
marker_middle         = '.'
color_end             = 'blue'
marker_end            = '.'


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

Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds, function)

Machine.fit(input_list, output_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("moddel 1")
for i in range(number_of_tested_neurons):

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 2+2*i))
    sys.stdout.flush()
    sleep(0.25)


    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = random_input_inner_for_player_A
    saved_random_input_inner_for_player_B = random_input_inner_for_player_B

    axarr[0][0].plot(np.round(Machine.processor(saved_random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(saved_random_input_inner_for_player_A), 2)[0], c = color_start, marker = marker_start , zorder=9999999)

    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr[0][0], color_middle, marker_middle)

    axarr[0][0].plot(np.round(Machine.processor(random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(random_input_inner_for_player_A), 2)[0], c = color_end, marker = marker_end, zorder=9999999 )


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


Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds,function)

Machine.fit(input_list, output_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


print("model 2")
for i in range(number_of_tested_neurons):

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 2+2*i))
    sys.stdout.flush()
    sleep(0.25)

    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = random_input_inner_for_player_A
    saved_random_input_inner_for_player_B = random_input_inner_for_player_B

    axarr[0][1].plot(np.round(Machine.processor(saved_random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(saved_random_input_inner_for_player_A), 2)[0], c = color_start, marker = marker_start , zorder =9999999)

    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr[0][1], color_middle, marker_middle)

    axarr[0][1].plot(np.round(Machine.processor(random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(random_input_inner_for_player_A), 2)[0], c = color_end,  marker = marker_end, zorder=9999999)

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


Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds, function)

Machine.fit(input_list, output_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("model 3")
for i in range(number_of_tested_neurons):

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 2+2*i))
    sys.stdout.flush()
    sleep(0.25)

    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = random_input_inner_for_player_A
    saved_random_input_inner_for_player_B = random_input_inner_for_player_B

    axarr[1][0].plot(np.round(Machine.processor(saved_random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(saved_random_input_inner_for_player_A), 2)[0], c = color_start , marker = marker_start, zorder =9999999)

    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr[1][0], color_middle, marker_middle)

    axarr[1][0].plot(np.round(Machine.processor(random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(random_input_inner_for_player_A), 2)[0], c = color_end, marker = marker_end, zorder=9999999)


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


Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds, function)

Machine.fit(input_list, output_list)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print('model 4')
for i in range(number_of_tested_neurons):

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 2+2*i))
    sys.stdout.flush()
    sleep(0.25)

    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 5 - 0
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = random_input_inner_for_player_A
    saved_random_input_inner_for_player_B = random_input_inner_for_player_B

    axarr[1][1].plot(np.round(Machine.processor(saved_random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(saved_random_input_inner_for_player_A), 2)[0], c = color_start , marker = marker_start, zorder =9999999)

    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr[1][1], color_middle, marker_middle)

    axarr[1][1].plot(np.round(Machine.processor(random_input_inner_for_player_B), 2)[0],np.round(Machine.processor(random_input_inner_for_player_A), 2)[0], c = color_end, marker = marker_end, zorder=9999999)

axarr[1][1].set_xlabel(r'$player \, B \, strategy$', fontsize=10)
axarr[1][1].set_xlim([0, 1])
axarr[1][1].set_ylim([1, 0])










plt.show()





