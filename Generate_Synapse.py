import numpy as np


#---------------------------------------------------------- Generating Samples -----------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-- This function generates a vector comprising of one-hotted vectors of 6 numbers to be used to train the neural network.
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_sudoku_array_one_hot(table_size, payoff_mode):

    if payoff_mode == "prewarm":

        sudoku_matrix_one_hot = np.zeros((table_size, table_size))
        for i in range(sudoku_matrix_one_hot.shape[0]):
            random_number = np.random.randint(table_size + 1)
            if random_number != table_size:
                sudoku_matrix_one_hot[i][random_number] = 1

        accept = False
        if np.count_nonzero(sudoku_matrix_one_hot == 1) < table_size:
            for j in range(sudoku_matrix_one_hot.shape[1]):
                if np.count_nonzero(sudoku_matrix_one_hot[:, j] == 1) > 1:
                    accept = True
        else:
            accept = True

        while accept == False:

            sudoku_matrix_one_hot = np.zeros((table_size, table_size))
            for i in range(sudoku_matrix_one_hot.shape[0]):
                random_number = np.random.randint(table_size + 1)
                if random_number != table_size:
                    sudoku_matrix_one_hot[i][random_number] = 1

            accept = False
            if np.count_nonzero(sudoku_matrix_one_hot == 1) < table_size:
                for j in range(sudoku_matrix_one_hot.shape[1]):
                    if np.count_nonzero(sudoku_matrix_one_hot[:, j] == 1) > 1:
                        accept = True
            else:
                accept = True

    if payoff_mode == "silent":

        sudoku_matrix_one_hot = np.zeros((table_size, table_size))
        for i in range(sudoku_matrix_one_hot.shape[0]):
            random_number = np.random.randint(table_size)
            sudoku_matrix_one_hot[i][random_number] = 1

    if payoff_mode == "spoil":

        sudoku_matrix_one_hot = np.zeros((table_size, table_size))
        for i in range(sudoku_matrix_one_hot.shape[0]):
            random_number = np.random.randint(table_size)
            sudoku_matrix_one_hot[i][random_number] = 1

        accept = True
        for j in range(sudoku_matrix_one_hot.shape[1]):
            if np.count_nonzero(sudoku_matrix_one_hot[:, j] == 1) != 1:
                accept = False

        while accept == False:

            sudoku_matrix_one_hot = np.zeros((table_size, table_size))
            for i in range(sudoku_matrix_one_hot.shape[0]):
                random_number = np.random.randint(table_size)
                sudoku_matrix_one_hot[i][random_number] = 1

            accept = True
            for j in range(sudoku_matrix_one_hot.shape[1]):
                if np.count_nonzero(sudoku_matrix_one_hot[:, j] == 1) != 1:
                    accept = False

    return sudoku_matrix_one_hot.flatten()


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-- This function generates the corresponding payoff vector for the vectors being generated by sudoku_array_one_hot function.
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def return_value(sudoku_array_one_hot, table_size):
    sudoku_matrix_one_hot = sudoku_array_one_hot.reshape((table_size, table_size))

    for j in range(sudoku_matrix_one_hot.shape[1]):
        if np.count_nonzero(sudoku_matrix_one_hot[:, j] == 1) != 1:
            returned        = np.zeros(table_size)
            returned[0:1] = 1
            return returned

    returned          = np.ones(table_size)
    return returned


table_size  = 6                   #<<<<<<<<<<< This element refers to the size of table to be sovled at hand.

payoff_mode = "silent"             #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

input_list  = list()
output_list = list()

for i in range(2000000):           #<<<<<<<<< This numbers decides the size of the samples being selected
    sudoku_array_one_hot       = generate_sudoku_array_one_hot(table_size, payoff_mode)
    sudoku_array_one_hot_value = return_value(sudoku_array_one_hot, table_size)
    input_list.append(sudoku_array_one_hot)
    output_list.append(sudoku_array_one_hot_value)


#----------------------------------------------------------Importing Model -----------------------------------------------------------------


from Brain_fast_6x6 import *       #<<<<<<<<< This imports the model of a neural network.

dims             = np.array([table_size * table_size, 100, 100, 100, table_size])  #<<<<<<<<< This element refers to the size or topology of the neural network.

tilt_1           = 30               #<<<<<<<<< This element refers to the intial slope for the activation functions in the hidden and output layers.
variation_1      = 0.1              #<<<<<<<<< This element refers to the range of randomness of the intial slopes of the activation functions in thehidden and output layers.
update_rate_1    = 0.000001         #<<<<<<<<< This element refers to learningg rate, identical to alpha.

variation_W      = 0.1              #<<<<<<<<<< This element refers to range of randomness of the initial weight matrix
update_rate_W    = 0.000001         #<<<<<<<<<< This element refers to learning rate, identical to alpha.

epochs           = 100000000        #<<<<<<<<<< This element refers to times of stachostic gradient decent.

tilt_2           = 0
variation_2      = 0
update_rate_2    = 0

deviation_inner  = 0
variation_inner  = 0
update_rate_inner= 0

rounds           = 0

mandatory_pulse  = 0
threshold        = 0.0

Machine          = Brain(dims, tilt_1, variation_1, update_rate_1, variation_W, update_rate_W, epochs, update_rate_2, update_rate_inner, rounds)


#----------------------------------------------------------Training by Model -----------------------------------------------------------------


retrain = False   #<<<<<<<<< This element decides whether to train weight matrix upon existing weight matrix

if retrain == True:

    Machine.synapse_list   = np.load("self.silent_2m_synapse_list_6x6_100x100x00_50_0.1_0.1_0.0000001_800m.npy")          #<<<<<<<<< This element imports the trained synapses for the neural network.
    Machine.tilt_1_list    = np.load("self.silent_2m_tilt_1_list_6x6_100x100x100_50_0.1_0.1_0.0000001_800m.npy")           #<<<<<<<<< This element imports the trained tilt_1 list for the neural network.
Machine.fit(input_list, output_list)


#----------------------------------------------------------Saving Synapses -----------------------------------------------------------------


np.save("self.silent_2m_synapse_list_6x6_100x100x00_30_0.1_0.1_0.000001_100m"             , Machine.synapse_list        )        #<<<<<<<<< This element exports the trained synapses for the neural network.
np.save("self.silent_2m_tilt_1_list_6x6_100x100x100_30_0.1_0.1_0.000001_100m"              , Machine.tilt_1_list         )        #<<<<<<<<< This element exports the trained tilt_1 list for the neural network.




