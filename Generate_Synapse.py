import numpy as np

#---------------------------------------------------------- Generating Samples -----------------------------------------------------------------

def generate_sodoku_array_one_hot(table_size):

    sodoku_matrix_one_hot = np.zeros((table_size, table_size))
    for i in range(sodoku_matrix_one_hot.shape[0]):
        random_number = np.random.randint(table_size + 1)
        if random_number != table_size:
            sodoku_matrix_one_hot[i][random_number] = 1

    accept = True
    if np.count_nonzero(sodoku_matrix_one_hot == 1) < table_size:
        accept = False
        for i in range(sodoku_matrix_one_hot.shape[0]):
            for j in range(sodoku_matrix_one_hot.shape[0]):
                if i != j:
                    if (np.amax(sodoku_matrix_one_hot[i]) == 1) & (np.amax(sodoku_matrix_one_hot[j]) == 1) & (np.argmax(sodoku_matrix_one_hot[i]) == np.argmax(sodoku_matrix_one_hot[j])):
                        accept == True


    while accept == False:
        sodoku_matrix_one_hot = np.zeros((table_size, table_size))
        for i in range(sodoku_matrix_one_hot.shape[0]):
            random_number = np.random.randint(table_size + 1)
            if random_number != table_size:
                sodoku_matrix_one_hot[i][random_number] = 1

        accept = True
        if np.count_nonzero(sodoku_matrix_one_hot == 1) < table_size:
            accept = False
            for i in range(sodoku_matrix_one_hot.shape[0]):
                for j in range(sodoku_matrix_one_hot.shape[0]):
                    if i != j:
                        if (np.amax(sodoku_matrix_one_hot[i]) == 1) & (np.amax(sodoku_matrix_one_hot[j]) == 1) & (np.argmax(sodoku_matrix_one_hot[i]) == np.argmax(sodoku_matrix_one_hot[j])):
                            accept == True


    return sodoku_matrix_one_hot.flatten()

def return_value(sodoku_array_one_hot, table_size):
    sodoku_matrix_one_hot = sodoku_array_one_hot.reshape((table_size, table_size))

    for i in range(sodoku_matrix_one_hot.shape[0]):
        for j in range(sodoku_matrix_one_hot.shape[0]):
            if i != j:
                if (np.amax(sodoku_matrix_one_hot[i]) == 1) & (np.amax(sodoku_matrix_one_hot[j]) == 1) & (np.argmax(sodoku_matrix_one_hot[i]) == np.argmax(sodoku_matrix_one_hot[j])):
                    returned        = np.zeros(table_size)
                    returned[0:1] = 1
                    return returned

    returned          = np.ones(table_size)
    return returned


table_size  = 6                                                                                               #<------------------------------------------------------

input_list  = list()
output_list = list()

for i in range(2000000):                                                                                            #<------------------------------------------------------
    sodoku_array_one_hot       = generate_sodoku_array_one_hot(table_size)
    sodoku_array_one_hot_value = return_value(sodoku_array_one_hot, table_size)
    input_list.append(sodoku_array_one_hot)
    output_list.append(sodoku_array_one_hot_value)

#----------------------------------------------------------Importing Model -----------------------------------------------------------------

from Brain_fast_6x6 import *                                                                                  #<------------------------------------------------------

dims       = np.array([table_size * table_size, 100, 100, 100, table_size])                                   #<------------------------------------------------------

alpha            = 0.000001                                                                                          #<------------------------------------------------------
epochs           = 400000000                                                                                           #<------------------------------------------------------

tilt_1           = 30                                                                                   #<------------------------------------------------------
variation_1      = 0.000001                                                                                    #<------------------------------------------------------
update_rate_1    = 0.000001                                                                                          #<---------------------------------------------------

beta             = 0.1
rounds           = 125

tilt_2           = 25
variation_2      = 0.1
update_rate_2    = 0.1

Machine          = Brain(dims, alpha, epochs, tilt_1, variation_1, update_rate_1, beta, rounds, update_rate_2)

#----------------------------------------------------------Training by Model -----------------------------------------------------------------


retrain = False                                                                                       #<------------------------------------------------------

if retrain == True:

    Machine.synapse_list   = np.load("self.synapse_list_6x6_100x100x100_0.000001_400m_tilt_30.npy")                #<------------------------------------------------------
    Machine.tilt_1_list    = np.load("self.tilt_1_list_6x6_100x100x100_0.000001_400m_tilt_30.npy")                #<------------------------------------------------------

Machine.fit(input_list, output_list)


#----------------------------------------------------------Saving Synapses -----------------------------------------------------------------

np.save("self.synapse_list_6x6_100x100x100_0.000001_400m_tilt_30"             , Machine.synapse_list        )        #<------------------------------------------------------
np.save("self.tilt_1_list_6x6_100x100x100_0.000001_400m_tilt_30"               , Machine.tilt_1_list         )        #<------------------------------------------------------




