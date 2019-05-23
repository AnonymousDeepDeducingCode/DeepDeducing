import numpy as np
from scipy.special import expit

#---------------------------------------------------------- Defining Needed Functions -----------------------------------------------------------------

#
#
#
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


def generate_sodoku_table(answer_matrix, dropped_numbers):
    size          = answer_matrix.shape[0]
    answer_matrix = answer_matrix.flatten()

    for i in range(dropped_numbers):
        random_index = np.random.randint(answer_matrix.shape[0])
        while answer_matrix[random_index] == 0:
            random_index = np.random.randint(answer_matrix.shape[0])
        answer_matrix[random_index] = 0

    answer_matrix = answer_matrix.reshape((size, size))

    return answer_matrix


def return_inner(sodoku_matrix, deviation, variation ):
    shape = sodoku_matrix.shape[0]
    sodoku_matrix_inner = np.zeros((shape ,shape , shape ))

    for i in range(sodoku_matrix.shape[0]):
        for j in range(sodoku_matrix.shape[1]):
            if sodoku_matrix[i][j] == 0:
                array = (np.random.random(shape) - 0.5) * variation  + deviation
                sodoku_matrix_inner[i][j] = array
            else:
                array = np.zeros(shape) - 15
                np.put(array, sodoku_matrix[i][j] - 1, 15)
                sodoku_matrix_inner[i][j] = array

    return sodoku_matrix_inner


def return_resistor(sodoku_matrix):
    shape = sodoku_matrix.shape[0]
    sodoku_matrix_inner = np.zeros((shape ,shape , shape ))
    for i in range(sodoku_matrix.shape[0]):
        for j in range(sodoku_matrix.shape[1]):
            if sodoku_matrix[i][j] == 0:
                array = np.zeros(shape) + 1
                sodoku_matrix_inner[i][j] = array
            else:
                array = np.zeros(shape)
                sodoku_matrix_inner[i][j] = array
    return sodoku_matrix_inner


def last_check(sodoku_matrix):
    for i in range(sodoku_matrix.shape[0]):
        for j in range(sodoku_matrix.shape[1]):
            target_number = sodoku_matrix[i][j]
            if (np.count_nonzero(sodoku_matrix[i, :] == target_number) != 1) |  (np.count_nonzero(sodoku_matrix[:, j] == target_number) != 1)  :
                return 0
    return 1


def mandatory_pulsed(sodoku_matrix_inner, tilt_2_list, sodoku_matrix_resistor, tilt_2, variation_2, deviation, variation, threshold):
    if np.count_nonzero(sodoku_matrix_resistor) != 0:
        i, j, k = np.where(sodoku_matrix_inner * tilt_2_list + sodoku_matrix_resistor * 100000 == np.amax(sodoku_matrix_inner * tilt_2_list + sodoku_matrix_resistor * 100000))
        i = i[0]
        j = j[0]
        k = k[0]

        if expit(sodoku_matrix_inner[i, j, k] * tilt_2_list[i, j, k]) >= (threshold):

            print(np.round(expit(sodoku_matrix_inner * tilt_2_list), 2))

            sodoku_matrix_inner[i, j] = sodoku_matrix_inner[i, j] * 0 - 15
            sodoku_matrix_inner[i, j, k] = 15
            sodoku_matrix_resistor[i, j] = sodoku_matrix_resistor[i, j] * 0

            for m in range(sodoku_matrix_inner.shape[0]):
               for n in range(sodoku_matrix_inner.shape[1]):
                   if (sodoku_matrix_resistor[m, n][0] == 1) :
                       sodoku_matrix_inner[m, n] = sodoku_matrix_inner[m, n] * variation  + deviation
                       tilt_2_list[m, n]         = (np.random.random((sodoku_matrix_inner.shape[2], )) -0.5 ) * variation_2 + tilt_2


    return sodoku_matrix_inner, tilt_2_list, sodoku_matrix_resistor

#---------------------------------------------------------- Starting Trials -----------------------------------------------------------------



number_of_trails = 100
score            = 0

for form in range(number_of_trails):

    table_size      = 6                                                                                           #<------------------------------------------------------

    dropped_numbers = 14                                                                                           #<------------------------------------------------------

    answer_matrix   = generate_answer_table(table_size, table_size)

    sodoku_matrix   = generate_sodoku_table(answer_matrix, dropped_numbers)


    #----------------------------------------------------------Importing Model -----------------------------------------------------------------

    from Brain_fast_6x6 import *                                                                         #<------------------------------------------------------

    dims             = np.array([table_size * table_size, 1, 1, 1, table_size])                             #<------------------------------------------------------

    alpha            = 0.0001
    epochs           = 2500000

    tilt_1           = 20
    variation_1      = 0.1
    update_rate_1    = 0.1

    deviation        = -0.1000                                                                                        #<------------------------------------------------------
    variation        = 0.0000                                                                                        #<------------------------------------------------------

    beta             = 0.0001                                                                                         #<------------------------------------------------------
    rounds           = 50000                                                                                                #<----------------------------------------------------

    tilt_2           = 30                                                                                             #<------------------------------------------------------
    variation_2      = 0.0000                                                                                       #<------------------------------------------------------
    update_rate_2    = 0.0001                                                                                       #<------------------------------------------------------


    mandatory_pulse  = 14                                                                           #<------------------------------------------------------
    threshold         = 0.0


    Machine          = Brain(dims, alpha, epochs, tilt_1, variation_1, update_rate_1, beta, rounds, update_rate_2)

    #----------------------------------------------------------Loading Synapses -----------------------------------------------------------------

    model   = "self.synapse_list_6x6_100x100x100_0.000001_400m_tilt_30.npy"                                                          # <------------------------------------------------------
    tilt_1  = "self.tilt_1_list_6x6_100x100x100_0.000001_400m_tilt_30.npy"                                                     # <------------------------------------------------------

    Machine.synapse_list                       = np.load(model,  allow_pickle=True)
    Machine.tilt_1_list                        = np.load(tilt_1, allow_pickle=True)

    print("---------------Model used-----------------")
    print(model)
    print(tilt_1)

    #----------------------------------------------------------Deducing by Model -----------------------------------------------------------------

    sodoku_matrix_inner    = return_inner(sodoku_matrix, deviation, variation )
    sodoku_matrix_resistor = return_resistor(sodoku_matrix)
    tilt_2_list            = (np.random.random((sodoku_matrix_inner.shape[0], sodoku_matrix_inner.shape[1], sodoku_matrix_inner.shape[2])) -0.5 ) * variation_2 + tilt_2


    goal                   = np.ones(table_size)

    # ---------------------------------------------------------------------- Mandatory Pulse -----------------------------------------------------------------------------------------------------------

    print(answer_matrix)

    print(sodoku_matrix)

    for i in range(mandatory_pulse):

        print(i)

        sodoku_matrix_inner, tilt_2_list                           = Machine.deduce_from(sodoku_matrix_inner, tilt_2_list, sodoku_matrix_resistor, goal)
        sodoku_matrix_inner, tilt_2_list, sodoku_matrix_resistor   = mandatory_pulsed(sodoku_matrix_inner, tilt_2_list, sodoku_matrix_resistor, tilt_2, variation_2, deviation, variation, threshold)

        machine_matrix = np.zeros((sodoku_matrix_inner.shape[0], sodoku_matrix_inner.shape[1]))
        for i in range(sodoku_matrix_inner.shape[0]):
            for j in range(sodoku_matrix_inner.shape[1]):
                if (sodoku_matrix_resistor[i][j][0] == 0):
                    machine_matrix[i][j] = np.argmax(sodoku_matrix_inner[i][j] * tilt_2_list[i][j]) + 1
        print(machine_matrix)


    print("---------------Final inners and lower tilts-----------------")

    print(np.round(sodoku_matrix_inner, 2))
    print("\n")
    print(np.round(tilt_2_list        , 2))

    print("--------------------------------")

    print(answer_matrix)

    print(sodoku_matrix)

    machine_matrix = sodoku_matrix
    for i in range(sodoku_matrix.shape[0]):
        for j in range(sodoku_matrix.shape[1]):
            machine_matrix[i][j] = np.argmax(sodoku_matrix_inner[i][j] * tilt_2_list[i][j]) + 1
    print(machine_matrix)

    print(answer_matrix - machine_matrix)

    print('--------------------------------Form: ', form + 1)

    if last_check(machine_matrix) == 1:
        score += 1

    print('--------------------------------Present score: ', score)

    f = open("result_small_6.txt", "a+")
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


