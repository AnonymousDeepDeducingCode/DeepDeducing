import numpy as np
from scipy.special import expit


#---------------------------------------------------------- Defining Needed Functions ---------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-- This function generates an answer table, a part of the numbers in which will be later dropped randomly to generate a Sudoku table
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_answer_table(row, column):
    mother_board = np.zeros((row, column))

    for i in range(row):
        for j in range(column):
            random_number      = np.random.randint(row) + 1
            fail = 0
            while ((random_number in mother_board[i, :]) )        | ((random_number in mother_board[:, j]) )     :
                random_number = np.random.randint(row) + 1
                fail += 1

                if fail >= 200:

                    return generate_answer_table(row, column)

            mother_board[i][j] = random_number

    return mother_board

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-- This function generates a Sudoku table with missing numbers
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_sudoku_table(answer_matrix, dropped_numbers):
    size          = answer_matrix.shape[0]
    answer_matrix = answer_matrix.flatten()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # -- This iteration randomly sets numbers to 0 in an answer table
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    for i in range(dropped_numbers):
        random_index = np.random.randint(answer_matrix.shape[0])
        while answer_matrix[random_index] == 0:
            random_index = np.random.randint(answer_matrix.shape[0])
        answer_matrix[random_index] = 0

    answer_matrix = answer_matrix.reshape((size, size))

    return answer_matrix

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-- This function return the inner values matrix for the missing and visible numbers in the Soduku table for the start of deducing
#----------------------------------------------------------------------------------------------------------------------------------------------------------
def return_inner(sudoku_matrix, deviation_inner, variation ):
    shape = sudoku_matrix.shape[0]
    sudoku_matrix_inner = np.zeros((shape ,shape , shape ))

    for i in range(sudoku_matrix.shape[0]):
        for j in range(sudoku_matrix.shape[1]):
            if sudoku_matrix[i][j] == 0:
                array = (np.random.random(shape) - 0.5) * variation  + deviation_inner
                sudoku_matrix_inner[i][j] = array
            else:
                array = np.zeros(shape) - 15
                np.put(array, sudoku_matrix[i][j] - 1, 15)
                sudoku_matrix_inner[i][j] = array

    return sudoku_matrix_inner

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-- This function generates a resistor matrix to make sure that, in the deducing phase, only the inner values for the missing numbers will be
#-- updated, while keeping the inner values of the visible number from being updated.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
def return_resistor(sudoku_matrix):
    shape = sudoku_matrix.shape[0]
    sudoku_matrix_inner = np.zeros((shape ,shape , shape ))
    for i in range(sudoku_matrix.shape[0]):
        for j in range(sudoku_matrix.shape[1]):
            if sudoku_matrix[i][j] == 0:
                array = np.zeros(shape) + 1
                sudoku_matrix_inner[i][j] = array
            else:
                array = np.zeros(shape)
                sudoku_matrix_inner[i][j] = array
    return sudoku_matrix_inner

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-- This function simply checks if the machine sovles the Sudoku table successfully. If successful, it returns 1, otherwise 0.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
def last_check(sudoku_matrix):
    for i in range(sudoku_matrix.shape[0]):
        for j in range(sudoku_matrix.shape[1]):
            target_number = sudoku_matrix[i][j]
            if (np.count_nonzero(sudoku_matrix[i, :] == target_number) != 1) |  (np.count_nonzero(sudoku_matrix[:, j] == target_number) != 1)  :
                return 0
    return 1

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#-- This function is the mandatory pulse function, which finds out the amax of the inner values of the missing numbers, solves a new
#-- number and re-initializes the rest of the inner values of the missing numbers  after the end of rounds.
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
def mandatory_pulsed(sudoku_matrix_inner, tilt_2_list, sudoku_matrix_resistor, tilt_2, variation_2, deviation_inner, variation, wash, threshold):
    if np.count_nonzero(sudoku_matrix_resistor) != 0:
        i, j, k = np.where(sudoku_matrix_inner * tilt_2_list + sudoku_matrix_resistor * 10000000 == np.amax(sudoku_matrix_inner * tilt_2_list + sudoku_matrix_resistor * 10000000))
        i = i[0]
        j = j[0]
        k = k[0]

        if expit(sudoku_matrix_inner[i, j, k] * tilt_2_list[i, j, k]) >= (threshold):

            print("---------------The output of the present inner values of the missing and visible numbers------------------")
            print(np.round(expit(sudoku_matrix_inner * tilt_2_list), 2))

            sudoku_matrix_inner[i, j] = sudoku_matrix_inner[i, j] * 0 - 15
            sudoku_matrix_inner[i, j, k] = 15
            sudoku_matrix_resistor[i, j] = sudoku_matrix_resistor[i, j] * 0

            if wash == True:
                for m in range(sudoku_matrix_inner.shape[0]):
                   for n in range(sudoku_matrix_inner.shape[1]):
                       if (sudoku_matrix_resistor[m, n][0] == 1) :
                           sudoku_matrix_inner[m, n] = sudoku_matrix_inner[m, n] * variation  + deviation_inner
                           tilt_2_list[m, n]         = (np.random.random((sudoku_matrix_inner.shape[2], )) -0.5 ) * variation_2 + tilt_2

    return sudoku_matrix_inner, tilt_2_list, sudoku_matrix_resistor


#---------------------------------------------------------- Starting Trials -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



number_of_trails = 100   #<<<<<<<<< This element refers to the number of tests or trials to be run to test if Deep Deducing can solve Sudoku tables.

score            = 0

for form in range(number_of_trails):

    table_size      = 6  #<<<<<<<<<<< This element refers to the size of table to be sovled at hand.

    dropped_numbers = 35  #<<<<<<<<< This element refers to the amount of numbers being dropped in an answer table, e.g. 35 dropped is equivaluent to 1 given.

    answer_matrix   = generate_answer_table(table_size, table_size)

    sudoku_matrix   = generate_sudoku_table(answer_matrix, dropped_numbers)


    #----------------------------------------------------------Importing Model ----------------------------------------------------------------------------------------------------------------------------------------------------


    from Brain_fast_6x6 import *       #<<<<<<<<< This imports the model of a neural network.

    dims             = np.array([table_size * table_size, 100, 100, 100, table_size])  #<<<<<<<<< This element refers to the size or topology of the neural network.

    tilt_1           = 0
    variation_1      = 0
    update_rate_1    = 0

    variation_W      = 0
    update_rate_W    = 0

    epochs           = 0

    tilt_2           = 30              #<<<<<<<<< This element refers to the intial slopes of the activation functions in the input layer
    variation_2      = 0.0001          #<<<<<<<<< This element refers to the range of randomness for the intial slopes of the activation functions in the input layer
    update_rate_2    = 0.0001          #<<<<<<<<< This element refers to deducing rate, identical to beta.

    deviation_inner  = -0.10000        #<<<<<<<<< This element refers to the initialized inner values for the missing numbers.
    variation_inner  = 0.0001          #<<<<<<<<< This element refers to the range of randomness for the initialized  inner values of the missing numbers.
    update_rate_inner= 0.0001          #<<<<<<<<<< This element refers to deducing rate, identical to beta.

    rounds           = 100000          #<<<<<<<<< This element refers to times of deducing before mandatory pulse takes place.

    mandatory_pulse  = 35              #<<<<<<<<< This element refers times of mandatory pulse, which is identical to dropped_numbers.
    threshold        = 0.0             #<<<<<<<<< This element determine the threhold which the amax of the inner values of the missing numbers must pass in order to initiate mandatory pulse
    wash             = True            #<<<<<<<<< This element determines whether the inner value of the rest of the missing numbers will be reset or not.

    Machine          = Brain(dims, tilt_1, variation_1, update_rate_1, variation_W, update_rate_W, epochs, update_rate_2, update_rate_inner, rounds)


    #----------------------------------------------------------Loading Synapses -----------------------------------------------------------------


    model   = "self.silent_2m_synapse_list_6x6_100x100x100_30_0.1_0.1_0.000001_200m.npy" #<<<<<<<<< This element imports the trained synapses for the neural network.
    tilt_1  = "self.silent_2m_tilt_1_list_6x6_100x100x100_30_0.1_0.1_0.000001_200m.npy"   #<<<<<<<<< This element imports the trained tilt_1 list for the neural network.

    Machine.synapse_list                       = np.load(model,  allow_pickle=True)
    Machine.tilt_1_list                        = np.load(tilt_1, allow_pickle=True)

    print("---------------Model used------------------------")
    print(model)
    print(tilt_1)


    #----------------------------------------------------------Deducing by Model -----------------------------------------------------------------


    sudoku_matrix_inner    = return_inner(sudoku_matrix, deviation_inner, variation_inner )
    sudoku_matrix_resistor = return_resistor(sudoku_matrix)
    tilt_2_list            = (np.random.random((sudoku_matrix_inner.shape[0], sudoku_matrix_inner.shape[1], sudoku_matrix_inner.shape[2])) -0.5 ) * variation_2 + tilt_2


    goal                   = np.ones(table_size)


    # ---------------------------------------------------------------------- Mandatory Pulse -----------------------------------------------------------------------------------------------------------


    print("---------------The answer table------------------")
    print(answer_matrix)
    print("---------------The Sudoku table------------------")
    print(sudoku_matrix)

    for times in range(mandatory_pulse):


        sudoku_matrix_inner, tilt_2_list                           = Machine.deduce_from(sudoku_matrix_inner, tilt_2_list, sudoku_matrix_resistor, goal)
        sudoku_matrix_inner, tilt_2_list, sudoku_matrix_resistor   = mandatory_pulsed(sudoku_matrix_inner, tilt_2_list, sudoku_matrix_resistor, tilt_2, variation_2, deviation_inner, variation_inner, wash, threshold)

        print("---------------The present numbers solved by machine--------------------")
        machine_matrix = np.zeros((sudoku_matrix_inner.shape[0], sudoku_matrix_inner.shape[1]))
        for i in range(sudoku_matrix_inner.shape[0]):
            for j in range(sudoku_matrix_inner.shape[1]):
                if (sudoku_matrix_resistor[i][j][0] == 0):
                    machine_matrix[i][j] = np.argmax(sudoku_matrix_inner[i][j] * tilt_2_list[i][j]) + 1
        print(machine_matrix)

        print("---------------Times of mandatory pulse so far----------------------------")
        print(times + 1)

        if np.count_nonzero(sudoku_matrix_resistor) == 0:
            break

    print("---------------Final inners values and lower tilts---------------------------")

    print(np.round(sudoku_matrix_inner, 2))
    print("\n")
    print(np.round(tilt_2_list        , 2))

    print("---------------Final comparison------------------------------------------------------------------------------------------")
    print("---------------The answer table-------------------------------------------------------------------------------------------")
    print(answer_matrix)
    print("---------------The Sudoku table------------------------------------------------------------------------------------------")
    print(sudoku_matrix)
    print("---------------The final table proposed by the machine-----------------------------------------------------------------")
    machine_matrix = sudoku_matrix
    for i in range(sudoku_matrix.shape[0]):
        for j in range(sudoku_matrix.shape[1]):
            machine_matrix[i][j] = np.argmax(sudoku_matrix_inner[i][j] * tilt_2_list[i][j]) + 1
    print(machine_matrix)
    print("---------------The difference between the final table proposed by the machine and the answer table-----------------")
    print(answer_matrix - machine_matrix)

    print('--------------------------------------------------------------------------------------------------------------------------------Form: ', form + 1)

    if last_check(machine_matrix) == 1:
        score += 1

    print('--------------------------------------------------------------------------------------------------------------------------------Present score: ', score)

    f = open("result_1.txt", "a+")
    f.write(str(" result "))
    f.write("\n")
    f.write(str(machine_matrix))
    f.write("\n")
    f.write(str(" form "))
    f.write(str(form + 1))
    f.write(str(" score "))
    f.write(str(score))
    f.write(str("\n"))
    f.write(str("\n"))
    f.close()

print("Maximum score:")
print(number_of_trails)
print("Total score:")
print(score)


