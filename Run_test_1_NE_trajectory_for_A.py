import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(0)




number_of_tested_neurons = 1



from Basic_Research_Model_for_test_1_NE_trajectory_for_A import *

input_dim             = 1 + 1
output_dim            = 10 + 10

alpha                 = 0.1
epochs                = 1000                                                                                     #---------------------------

beta                  = 0.1
rounds                = 1000                                                                                      #---------------------------


function              = "sigmoid"

color_start           = 'red'
marker_start          = 's'
color_middle          =  ((238 ) / 255, (130) / 255, (238 ) / 255)
marker_middle         = '.'
color_end             = 'blue'
marker_end            = 's'

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

for i in range(number_of_tested_neurons):

    strategy_neurons_size             = 1
    random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 0 + 2
    desired_payoff_for_player_A       = 1
    random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 0 - 2
    desired_payoff_for_player_B       = 1

    saved_random_input_inner_for_player_A = copy.deepcopy(random_input_inner_for_player_A)
    saved_random_input_inner_for_player_B = copy.deepcopy(random_input_inner_for_player_B)





    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y)
    def z_of(x, y):
        z = np.zeros((100, 100))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                layer_0, \
                layer_1, \
                layer_2 = Machine.generate_values_for_each_layer(np.concatenate((np.array([y[i][j]]), np.array([x[i][j]]))))
                z[i][j] = np.sum((np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) - layer_2[0:10]) ** 2)
        return z
    z = z_of(x, y)
    ax.plot_surface(x, y, z, cmap='cool', alpha=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')



    layer_0, \
    layer_1, \
    layer_2 = Machine.generate_values_for_each_layer(np.concatenate((Machine.sigmoid(saved_random_input_inner_for_player_A), Machine.sigmoid(saved_random_input_inner_for_player_B))))

    start_of_B = Machine.processor(saved_random_input_inner_for_player_B)[0]
    start_of_A = Machine.processor(saved_random_input_inner_for_player_A)[0]
    ax.scatter(start_of_B, start_of_A, np.sum( (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) - layer_2[0:10] )**2), c=color_start, marker=marker_start)



    random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, ax, color_middle, marker_middle)

    layer_0, \
    layer_1, \
    layer_2 = Machine.generate_values_for_each_layer(np.concatenate((Machine.sigmoid(random_input_inner_for_player_A), Machine.processor(random_input_inner_for_player_B))))

    start_of_B = Machine.processor(random_input_inner_for_player_B)[0]
    start_of_A = Machine.processor(random_input_inner_for_player_A)[0]
    ax.scatter(start_of_B, start_of_A, np.sum( (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) - layer_2[0:10] )**2), c=color_end, marker=marker_end)



plt.xlim([0, 1])
plt.xlabel(r'$player \, B \, strategy$', fontsize=10)
plt.ylim([1, 0])
plt.ylabel(r'$player \, A \, strategy$', fontsize=10)
plt.title(r'$error \, surface \, for \, player \, A$', fontsize=10, y=1.08)
plt.show()

