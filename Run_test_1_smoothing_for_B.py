import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

np.random.seed(0)




number_of_tested_neurons = 1


from Basic_Research_Model_for_test_1_smoothing_for_B import *

input_dim             = 1 + 1
output_dim            = 10 + 10

alpha                 = 0.1
epochs                = 100000                                                                                     #---------------------------

beta                  = 0.1
rounds                = 1                                                                                      #---------------------------
                                                                            #---------------------------
function              = "sigmoid"


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

    fig = plt.figure()
    ax  = fig.gca(projection='3d')
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y)
    def z_of(x, y):
        z = np.zeros((100, 100))
        print(z.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                layer_0, \
                layer_1, \
                layer_2 = Machine.generate_values_for_each_layer(np.concatenate((np.array([y[i][j]]), np.array([x[i][j]]))))
                z[i][j] = np.sum((np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) - layer_2[-10:]) ** 2)
        return z
    z = z_of(x, y)
    ax.plot_surface(x, y, z, cmap='cool', alpha=0.7)
    deviation = 0.0
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

plt.xlim([0, 1])
plt.xlabel(r'$player \, B \, strategy$', fontsize=10)
plt.ylim([1, 0])
plt.ylabel(r'$player \, A \, strategy$', fontsize=10)
plt.title(r'$error \, surface \, for \, player \, B \, under \, training \, epochs\, of \, 10^5$', fontsize=10, y=1.08)
plt.show()

