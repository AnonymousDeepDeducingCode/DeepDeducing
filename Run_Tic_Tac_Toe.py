import numpy as np
import math
import copy


strategies_so_far = [np.array([0, 0, 0,
                               0, 0, 1,
                               0, 0, 0], dtype = float),
                     np.array([0, 0, 0,
                               0, 1, 0,
                               0, 0, 0], dtype = float),
                     np.array([0, 0, 1,
                               0, 0, 0,
                               0, 0, 0], dtype=float),
                     np.array([0, 0, 0,
                               0, 0, 0,
                               0, 0, 1], dtype=float),
                     np.array([1, 0, 0,
                               0, 0, 0,
                               0, 0, 0], dtype=float),
                     np.array([0, 1, 0,
                               0, 0, 0,
                               0, 0, 0], dtype=float),
                     np.array([0, 0, 0,
                               0, 0, 0,
                               0, 1, 0], dtype=float),
                     ]






def turning_strategies_into_matrix(strategies):
    strategies = np.array(strategies)
    matrix = np.zeros((3, 3))
    matrix = matrix.flatten()
    for i in  range(strategies.shape[0]):
        if i % 2 == 0:
            matrix[np.argmax(strategies[i])] = 1
        if i % 2 == 1:
            matrix[np.argmax(strategies[i])] = 2
    matrix = matrix.reshape((3, 3))
    return matrix

matrix = turning_strategies_into_matrix(strategies_so_far)

print("The current state of the board is:")
print(matrix)





def counting_remain_steps(matrix):
    matrix = matrix.flatten()
    number_of_total_zeros = 0
    for i in range(matrix.shape[0]):
        if matrix[i] == 0:
            number_of_total_zeros += 1
    return number_of_total_zeros

remain_steps      = counting_remain_steps(matrix)
print("The remaining steps are:")
print(remain_steps)
midterm_incentive = 0.0          # mid-term heuristic payoff
final_incentive   = 1            # payoff in the final stage
samples_selected  = 1000 * remain_steps






alpha             = 0.5
epochs            = 1000 * remain_steps
beta              = 0.5
rounds            = 20000


# odds for attack, while even for defence
if remain_steps % 2 == 0:
    steps_foreseen = remain_steps
if remain_steps % 2 == 1:
    steps_foreseen = remain_steps - 1
print("The foreseen steps are:")
print(steps_foreseen)


#================================= REGULATING THE RULES ==============================================================

def payoff_for_player_1(matrix, matrix_shape_0, midterm_incentive):
    payoff_for_player_1 = 0
    payoff_for_player_1_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i-1][j] == 1) & (matrix[i+1][j] == 0)) :

                        payoff_for_player_1 += midterm_incentive
                    elif ((matrix[i-1][j] == 0) & (matrix[i+1][j] == 1)):

                        payoff_for_player_1 += midterm_incentive

                    else:
                        payoff_for_player_1_smaller = 0



    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i][j-1] == 1) & (matrix[i][j+1] == 0)) | ((matrix[i][j-1] == 0) & (matrix[i][j+1] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i-1][j] == 1) & (matrix[i+1][j] == 1)) :

                        payoff_for_player_1 += midterm_incentive
                    elif ((matrix[i-1][j] == 1) & (matrix[i+1][j] == 1)):

                        payoff_for_player_1 += midterm_incentive

                    else:
                        payoff_for_player_1_smaller = 0



    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i][j-1] == 1) & (matrix[i][j+1] == 1)) | ((matrix[i][j-1] == 1) & (matrix[i][j+1] == 1)):
                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i-1][j-1] == 1) & (matrix[i+1][j+1] == 0)) :

                        payoff_for_player_1 += midterm_incentive
                    elif ((matrix[i-1][j-1] == 0) & (matrix[i+1][j+1] == 1)):

                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0



    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i+1][j-1] == 1) & (matrix[i-1][j+1] == 0)) :

                        payoff_for_player_1 += midterm_incentive
                    if ((matrix[i+1][j-1] == 0) & (matrix[i-1][j+1] == 1)):

                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i-1][j-1] == 1) & (matrix[i+1][j+1] == 1)) :

                        payoff_for_player_1 += midterm_incentive
                    elif ((matrix[i-1][j-1] == 1) & (matrix[i+1][j+1] == 1)):

                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0



    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i+1][j-1] == 1) & (matrix[i-1][j+1] == 1)) :

                        payoff_for_player_1 += midterm_incentive
                    if ((matrix[i+1][j-1] == 1) & (matrix[i-1][j+1] == 1)):

                        payoff_for_player_1 += midterm_incentive
                    else:
                        payoff_for_player_1_smaller = 0




    # ==================================================================================================================

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i-1][j] == 1) & (matrix[i+1][j] == 1)):
                        payoff_for_player_1 = 1

                    else:
                        payoff_for_player_1_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i][j-1] == 1) & (matrix[i][j+1] == 1)):
                        payoff_for_player_1 = 1

                    else:
                        payoff_for_player_1_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i-1][j-1] == 1) & (matrix[i+1][j+1] == 1)) :
                        payoff_for_player_1 = 1

                    else:
                        payoff_for_player_1_smaller = 0



    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 1:
                    if ((matrix[i+1][j-1] == 1) & (matrix[i-1][j+1] == 1)) :
                        payoff_for_player_1 = 1

                    else:
                        payoff_for_player_1_smaller = 0


    return np.amax(payoff_for_player_1, payoff_for_player_1_smaller)

def payoff_for_player_2(matrix, matrix_shape_0, midterm_incentive):
    payoff_for_player_2 = 0
    payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i-1][j] == 2) & (matrix[i+1][j] == 0)) :

                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i-1][j] == 0) & (matrix[i+1][j] == 2)):

                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i][j-1] == 2) & (matrix[i][j+1] == 0)) | ((matrix[i][j-1] == 0) & (matrix[i][j+1] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i-1][j] == 2) & (matrix[i+1][j] == 2)) :

                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i-1][j] == 2) & (matrix[i+1][j] == 2)):

                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i][j-1] == 2) & (matrix[i][j+1] == 2)) | ((matrix[i][j-1] == 0) & (matrix[i][j+1] == 2)):
                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i-1][j-1] == 2) & (matrix[i+1][j+1] == 0)) :

                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i-1][j-1] == 0) & (matrix[i+1][j+1] == 2)):

                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0



    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i+1][j-1] == 2) & (matrix[i-1][j+1] == 0)) :

                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i+1][j-1] == 0) & (matrix[i-1][j+1] == 2)):

                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i-1][j-1] == 2) & (matrix[i+1][j+1] == 2)) :

                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i-1][j-1] == 2) & (matrix[i+1][j+1] == 2)):

                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0



    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 0:
                    if ((matrix[i+1][j-1] == 2) & (matrix[i-1][j+1] == 2)) :

                        payoff_for_player_2 += midterm_incentive
                    if ((matrix[i+1][j-1] == 2) & (matrix[i-1][j+1] == 2)):

                        payoff_for_player_2 += midterm_incentive
                    else:
                        payoff_for_player_2_smaller = 0


    # ==================================================================================================================

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i-1][j] == 2) & (matrix[i+1][j] == 2)):
                        payoff_for_player_2 = 1
                    else:
                        payoff_for_player_2_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i][j-1] == 2) & (matrix[i][j+1] == 2)):
                        payoff_for_player_2 = 1
                    else:
                        payoff_for_player_2_smaller = 0


    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i-1][j-1] == 2) & (matrix[i+1][j+1] == 2)):
                        payoff_for_player_2 = 1
                    else:
                        payoff_for_player_2_smaller = 0



    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if ((j - 1) >= 0) & ((j + 1) <= matrix.shape[1] - 1) & ((i - 1) >= 0) & ((i + 1) <= matrix.shape[0] - 1):
                if matrix[i][j] == 2:
                    if ((matrix[i+1][j-1] == 2) & (matrix[i-1][j+1] == 2)) :
                        payoff_for_player_2 = 1
                    else:
                        payoff_for_player_2_smaller = 0



    return np.amax(payoff_for_player_2, payoff_for_player_2_smaller)

#================================= SIMULATING THE STRATEGIES AND OUTCOMES ==========================================

def vectorize_strategy(matrix, strategy_int):
    vector = np.zeros(matrix.shape[0] * matrix.shape[1])
    vector[strategy_int] = 1
    return vector

def put_vectorized_strategy_of_player_1_into_matrix(matrix, vectorized_strategy):

    return matrix + vectorized_strategy.reshape((3,3))

def put_vectorized_strategy_of_player_2_into_matrix(matrix, vectorized_strategy):

    return matrix + vectorized_strategy.reshape((3,3)) * 2

def check_full(matrix, strategy_index):
    if matrix.reshape(9)[strategy_index] != 0:
        return "full"

def generate_from(strategies_so_far, matrix, remain_steps, midterm_incentive, final_incentive, samples_selected):
    X = list()
    Y = list()
    matrix_initial = copy.deepcopy(matrix)
    np.random.seed(0)
    for i in range(samples_selected):

        strategies = copy.deepcopy(strategies_so_far)
        outcome    = list()
        matrix     = copy.deepcopy(matrix_initial)

        payoff     = 0

        for j in range(  remain_steps  ):



            if ((np.array(strategies_so_far).shape[0] + j + 1)% 2 == 1):

                if (payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff == np.array([0, final_incentive])).all() | (payoff == np.array([final_incentive, 0])).all() :

                    break

                else:

                    player_1_strategy = np.random.randint(matrix.shape[0] *  matrix.shape[1])
                    if check_full(matrix, player_1_strategy) == "full":
                        payoff = np.array([0, final_incentive])
                        outcome.append(payoff)

                    player_1_strategy = vectorize_strategy(matrix, player_1_strategy)
                    matrix            = put_vectorized_strategy_of_player_1_into_matrix(matrix, player_1_strategy)
                    strategies.append(player_1_strategy)

                    if  (payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) == 1) :

                        payoff = np.array([ final_incentive , 0   ])

                        outcome.append(payoff)



            if ((np.array(strategies_so_far).shape[0] + j + 1)% 2 == 0):

                if (payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff == np.array([0, final_incentive])).all() | (payoff == np.array([final_incentive, 0])).all() :

                    break

                else:

                    player_2_strategy = np.random.randint(matrix.shape[0] * matrix.shape[1])
                    if check_full(matrix, player_2_strategy) == "full":
                        payoff = np.array([final_incentive, 0])
                        outcome.append(payoff)

                    player_2_strategy = vectorize_strategy(matrix, player_2_strategy)
                    matrix            = put_vectorized_strategy_of_player_2_into_matrix(matrix, player_2_strategy)
                    strategies.append(player_2_strategy)

                    if  (payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive) == 1) :

                        payoff = np.array([ 0 , final_incentive  ])

                        outcome.append(payoff)

            if (j == remain_steps -1) :

                if (payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive) == 1) | (payoff == np.array([0, final_incentive])).all() | (payoff == np.array([final_incentive, 0])).all():

                    break

                else:

                    payoff = np.array([ payoff_for_player_1(matrix, matrix.shape[0], midterm_incentive) , payoff_for_player_2(matrix, matrix.shape[0], midterm_incentive)  ])

                    outcome.append(payoff)



        X.append(strategies)
        Y.append(outcome[0])

    return X, Y

#================================= GENERATING THE STRATEGIES AND OUTCOMES==========================================

X, Y = generate_from(strategies_so_far ,matrix,  remain_steps, midterm_incentive, final_incentive, samples_selected)

X    = np.array(X)
Y    = np.array(Y)

#================================ TRAINING THE MODEL =================================================================

from Game_Net_Tic_Tac_Toe import *

strategies_so_far = np.array(strategies_so_far)

Game              = Game_Net( 9  , alpha, epochs, beta, rounds, strategies_so_far, steps_foreseen)

Game.fit(X, Y)

#================================ TRAINING BY BACK  DEDUCTION =======================================================

Game.predict()

#================================ PRINTING OUT THE STRATEGY ==========================================================


#for i in range(steps_foreseen):
#    print('Current Player Best Strategy:')
#    print(Game.sigmoid(Game.strategies_foreseen[i]).reshape((3, 3)))



def turning_strategy_into_board(strategy, board):
    ones = np.zeros_like(board)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i][j] != 0:
                ones[i][j] = 1
    strategy_board = strategy.reshape((3, 3)) - strategy.reshape((3, 3)) * ones
    point = np.argmax(strategy_board)
    new_board = board.flatten()
    if np.sum(ones) % 2 == 0:
        new_board[point] = 1
    else:
        new_board[point] = 2
    return(new_board.reshape((3, 3)))

print("The final movement for Deep Deducing is:")
print(turning_strategy_into_board(Game.sigmoid(Game.strategies_foreseen[0]), matrix) )




