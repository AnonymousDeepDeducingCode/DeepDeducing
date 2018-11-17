import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
np.random.seed(0)

class Basic_Research_Model(object):
    def __init__(self, dims, alpha, epochs, beta, rounds,rounds_inside_swarm,  signal_receiver, signal_transfer, function):

        self.dims                         = dims

        self.number_of_layers             = self.dims.shape[0]

        self.alpha                        = alpha
        self.epochs                       = epochs
        self.beta                         = beta
        self.rounds                       = rounds
        self.rounds_inside_swarm          = rounds_inside_swarm

        self.signal_receiver              = signal_receiver
        self.signal_transfer              = signal_transfer
        self.function                     = function


        self.synapse_list                 = self.initialize_weights()

        self.synapse_list_update          = np.zeros_like( self.synapse_list)



    def initialize_weights(self):

        synapse_list = list()
        for i in range(self.number_of_layers - 1):
            synapse                                              = (np.random.random((self.dims[i]                                 , self.dims[i+1]                          )) -0.5 ) * 0.1
            synapse_list.append(synapse)
        synapse_list = np.array(synapse_list)
        return  synapse_list

    def life_signal_receiver(self, x):
        if self.signal_receiver == "tanh":
            output = np.tanh(x)
        if self.signal_receiver == "monotonic":
            output = x
        if self.signal_receiver == "sigmoid":
            output = 1 / (1 + np.exp(-x))
        if self.signal_receiver == "ReLu":
            output = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                if (x[i] >= 0) == True & (x[i] <= 1) == True:
                    output[i] = x[i]
                if (x[i]) > 1 == True:
                    output[i] = 1
                if (x[i]) < 0 == True:
                    output[i] = 0
        if self.signal_receiver == "Leaky ReLu":
            output = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                if (x[i] >= 0) == True & (x[i] <= 1) == True:
                    output[i] = x[i]
                if (x[i]) > 1 == True:
                    output[i] = x[i]
                if (x[i]) < 0 == True:
                    output[i] = 0.1 * x[i]
        return output

    def life_signal_receiver_output_to_derivative(self, output):
        if self.signal_receiver == "tanh":
            output = 1 - output * (output)
        if self.signal_receiver == "monotonic":
            output = 1
        if self.signal_receiver == "sigmoid":
            output = output * (1 - output)
        if self.signal_receiver == "ReLu":
            dummy = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                if (output[i] > 0) == True & (output[i] < 1) == True:
                    dummy[i] = 1
                if (output[i] >= 1) == True:
                    dummy[i] = 0
                if (output[i] <= 0) == True:
                    dummy[i] = 0
            output = dummy
        if self.signal_receiver == "Leaky ReLu":
            dummy = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                if (output[i] > 0) == True & (output[i] < 1) == True:
                    dummy[i] = 1
                if (output[i] >= 1) == True:
                    dummy[i] = 1
                if (output[i] <= 0) == True:
                    dummy[i] = 0.1
            output = dummy
        return output

    def sigmoid(self, x):
        if self.signal_transfer == "monotonic":
            output = x
        if self.signal_transfer == "sigmoid":
            output = 1/(1+np.exp(-x))
        if self.signal_transfer == "ReLu":
            output = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                if (x[i] >= 0) == True & (x[i] <= 1) == True:
                    output[i] = x[i]
                if (x[i]) > 1 == True:
                    output[i] = 1
                if (x[i]) < 0 == True:
                    output[i] = 0
        if self.signal_transfer == "Leaky ReLu":
            output = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                if (x[i] >= 0) == True & (x[i] <= 1) == True:
                    output[i] = x[i]
                if (x[i]) > 1 == True:
                    output[i] = x[i]
                if (x[i]) < 0 == True:
                    output[i] = 0.1 * x[i]
        return output

    def sigmoid_output_to_derivative(self, output):
        if self.signal_transfer == "monotonic":
            output = 1
        if self.signal_transfer == "sigmoid":
            output = output*(1-output)
        if self.signal_transfer == "ReLu":
            dummy = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                if (output[i] > 0) == True & (output[i] < 1) == True:
                    dummy[i] = 1
                if (output[i] >= 1) == True:
                    dummy[i] = 0
                if (output[i] <= 0) == True:
                    dummy[i] = 0
            output = dummy
        if self.signal_transfer == "Leaky ReLu":
            dummy = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                if (output[i] > 0) == True & (output[i] < 1) == True:
                    dummy[i] = 1
                if (output[i] >= 1) == True:
                    dummy[i] = 1
                if (output[i] <= 0) == True:
                    dummy[i] = 0.1
            output = dummy
        return output

    def processor(self, x):
        if self.function == "sigmoid":
            output = 1 / (1 + np.exp(-1 * x))
        if self.function == "ReLu":
            output = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                if (x[i] >= 0) == True & (x[i] <= 1) == True:
                    output[i] = x[i]
                if (x[i]) > 1 == True:
                    output[i] = 1
                if (x[i]) < 0 == True:
                    output[i] = 0
        if self.function == "Leaky ReLu":
            output = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                if (x[i] >= 0) == True & (x[i] <= 1) == True:
                    output[i] = x[i]
                if (x[i]) > 1 == True:
                    output[i] = x[i]
                if (x[i]) < 0 == True:
                    output[i] = 0.1 * x[i]
        return output

    def processor_output_to_derivative(self, output):
        if self.function == "sigmoid":
            output = output * (1 - output) * 1
        if self.function == "ReLu":
            dummy = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                if (output[i] > 0) == True & (output[i] < 1) == True:
                    dummy[i] = 1
                if (output[i] >= 1) == True:
                    dummy[i] = 0
                if (output[i] <= 0) == True:
                    dummy[i] = 0
            output = dummy
        if self.function == "Leaky ReLu":
            dummy = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                if (output[i] > 0) == True & (output[i] < 1) == True:
                    dummy[i] = 1
                if (output[i] >= 1) == True:
                    dummy[i] = 1
                if (output[i] <= 0) == True:
                    dummy[i] = 0.1
            output = dummy
        return output

    def generate_values_for_each_layer(self, input):

        layer_list                = list()

        layer                     = copy.deepcopy(np.array(input))

        layer_list.append(layer)

        for i in range(self.number_of_layers - 2):

            layer                 = self.sigmoid(np.dot(layer_list[i]                          , self.synapse_list[i]                                                          ) )

            layer_list.append(layer)

        layer                 = self.life_signal_receiver(np.dot(layer_list[-1]                          , self.synapse_list[-1]                                                          ) )

        layer_list.append(layer)

        return   np.array(copy.deepcopy(layer_list))

    def train_for_each(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 2):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)



        for i in range(self.number_of_layers - 1):

            self.synapse_list_update[i]                            += np.atleast_2d(layer_list[i]                                                          ).T.dot(layer_delta_list[- 1 - i]                              )

        self.synapse_list                                                 += self.synapse_list_update                   * self.alpha

        self.synapse_list_update                                          *= 0



    def fit(self, input_list, output_list):

        for i in range(self.epochs):

            random_index = np.random.randint(input_list.shape[0])
            input        = input_list[random_index]
            output       = output_list[random_index]

            layer_list = self.generate_values_for_each_layer(input)

            self.train_for_each(
                       layer_list,
                       output)

        return self

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def train_for_random_input_inner_player_A(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A                     += self.momentum_A * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A) * self.beta
        self.momentum_A                                           = self.momentum_A * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A) * self.beta





    def train_for_random_input_inner_player_A_2(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_2                   += self.momentum_A_2 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_2.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_2) * self.beta
        self.momentum_A_2                                         = self.momentum_A_2 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_2.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_2) * self.beta



    def train_for_random_input_inner_player_A_3(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_3                   += self.momentum_A_3 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_3.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_3) * self.beta
        self.momentum_A_3                                         = self.momentum_A_3 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_3.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_3) * self.beta


    def train_for_random_input_inner_player_A_4(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_4                   += self.momentum_A_4 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_4.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_4) * self.beta
        self.momentum_A_4                                         = self.momentum_A_4 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_4.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_4) * self.beta



    def train_for_random_input_inner_player_A_5(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_5                   += self.momentum_A_5 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_5.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_5) * self.beta
        self.momentum_A_5                                         = self.momentum_A_5 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_5.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_5) * self.beta



    def train_for_random_input_inner_player_A_6(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_6                   += self.momentum_A_6 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_6.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_6) * self.beta
        self.momentum_A_6                                         = self.momentum_A_6 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_6.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_6) * self.beta


    def train_for_random_input_inner_player_A_7(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_7                   += self.momentum_A_7 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_7.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_7) * self.beta
        self.momentum_A_7                                         = self.momentum_A_7 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_7.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_7) * self.beta

    def train_for_random_input_inner_player_A_8(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_8                   += self.momentum_A_8 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_8.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_8) * self.beta
        self.momentum_A_8                                         = self.momentum_A_8 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_8.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_8) * self.beta

    def train_for_random_input_inner_player_A_9(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_9                   += self.momentum_A_9 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_9.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_9) * self.beta
        self.momentum_A_9                                         = self.momentum_A_9 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_9.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_9) * self.beta


    def train_for_random_input_inner_player_A_10(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_10                   += self.momentum_A_10 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_10.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_10) * self.beta
        self.momentum_A_10                                         = self.momentum_A_10 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_10.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_10) * self.beta


    def train_for_random_input_inner_player_A_11(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_11                   += self.momentum_A_11 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_11.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_11) * self.beta
        self.momentum_A_11                                         = self.momentum_A_11 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_11.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_11) * self.beta



    def train_for_random_input_inner_player_A_12(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_12                   += self.momentum_A_12 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_12.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_12) * self.beta
        self.momentum_A_12                                         = self.momentum_A_12 * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_12.shape[0]] * self.beta + (self.chosen_A - self.random_input_inner_for_player_A_12) * self.beta


    def train_for_random_input_inner_player_A_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_A_                     = self.random_input_inner_for_player_A + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A.shape[0]] * self.beta




    def train_for_random_input_inner_player_A_2_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_2_                     = self.random_input_inner_for_player_A_2 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_2.shape[0]] * self.beta



    def train_for_random_input_inner_player_A_3_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_3_                     = self.random_input_inner_for_player_A_3 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_3.shape[0]] * self.beta



    def train_for_random_input_inner_player_A_4_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_4_                     = self.random_input_inner_for_player_A_4 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_4.shape[0]] * self.beta




    def train_for_random_input_inner_player_A_5_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_5_                     = self.random_input_inner_for_player_A_5 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_5.shape[0]] * self.beta



    def train_for_random_input_inner_player_A_6_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_6_                     = self.random_input_inner_for_player_A_6 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_6.shape[0]] * self.beta



    def train_for_random_input_inner_player_A_7_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_7_                     = self.random_input_inner_for_player_A_7 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_7.shape[0]] * self.beta




    def train_for_random_input_inner_player_A_8_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_8_                     = self.random_input_inner_for_player_A_8 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_8.shape[0]] * self.beta




    def train_for_random_input_inner_player_A_9_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_9_                     = self.random_input_inner_for_player_A_9 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_9.shape[0]] * self.beta




    def train_for_random_input_inner_player_A_10_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_10_                     = self.random_input_inner_for_player_A_10 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_10.shape[0]] * self.beta




    def train_for_random_input_inner_player_A_11_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_11_                     = self.random_input_inner_for_player_A_11 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_11.shape[0]] * self.beta





    def train_for_random_input_inner_player_A_12_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)


        self.random_input_inner_for_player_A_12_                     = self.random_input_inner_for_player_A_12 + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A_12.shape[0]] * self.beta



    def train_for_random_input_inner_player_B(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B                     += self.momentum_B * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B) * self.beta
        self.momentum_B                                           = self.momentum_B * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B) * self.beta




    def train_for_random_input_inner_player_B_2(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_2                   += self.momentum_B_2 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_2.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_2) * self.beta
        self.momentum_B_2                                         = self.momentum_B_2 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_2.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_2) * self.beta



    def train_for_random_input_inner_player_B_3(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_3                   += self.momentum_B_3 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_3.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_3) * self.beta
        self.momentum_B_3                                         = self.momentum_B_3 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_3.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_3) * self.beta



    def train_for_random_input_inner_player_B_4(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_4                   += self.momentum_B_4 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_4.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_4) * self.beta
        self.momentum_B_4                                         = self.momentum_B_4 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_4.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_4) * self.beta



    def train_for_random_input_inner_player_B_5(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_5                   += self.momentum_B_5 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_5.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_5) * self.beta
        self.momentum_B_5                                         = self.momentum_B_5 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_5.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_5) * self.beta


    def train_for_random_input_inner_player_B_6(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_6                   += self.momentum_B_6 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_6.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_6) * self.beta
        self.momentum_B_6                                         = self.momentum_B_6 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_6.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_6) * self.beta


    def train_for_random_input_inner_player_B_7(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_7                   += self.momentum_B_7 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_7.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_7) * self.beta
        self.momentum_B_7                                         = self.momentum_B_7 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_7.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_7) * self.beta


    def train_for_random_input_inner_player_B_8(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_8                   += self.momentum_B_8 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_8.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_8) * self.beta
        self.momentum_B_8                                         = self.momentum_B_8 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_8.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_8) * self.beta


    def train_for_random_input_inner_player_B_9(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_9                   += self.momentum_B_9 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_9.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_9) * self.beta
        self.momentum_B_9                                         = self.momentum_B_9 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_9.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_9) * self.beta

    def train_for_random_input_inner_player_B_10(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_10                   += self.momentum_B_10 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_10.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_10) * self.beta
        self.momentum_B_10                                         = self.momentum_B_10 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_10.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_10) * self.beta


    def train_for_random_input_inner_player_B_11(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_11                   += self.momentum_B_11 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_11.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_11) * self.beta
        self.momentum_B_11                                         = self.momentum_B_11 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_11.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_11) * self.beta


    def train_for_random_input_inner_player_B_12(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_12                   += self.momentum_B_12 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_12.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_12) * self.beta
        self.momentum_B_12                                         = self.momentum_B_12 * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_12.shape[0]: ] * self.beta + (self.chosen_B - self.random_input_inner_for_player_B_12) * self.beta





    def train_for_random_input_inner_player_B_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_                     =  self.random_input_inner_for_player_B + layer_delta_list[-1][0][-self.random_input_inner_for_player_B.shape[0]: ] * self.beta



    def train_for_random_input_inner_player_B_2_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_2_                     =  self.random_input_inner_for_player_B_2 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_2.shape[0]: ] * self.beta



    def train_for_random_input_inner_player_B_3_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_3_                     =  self.random_input_inner_for_player_B_3 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_3.shape[0]: ] * self.beta


    def train_for_random_input_inner_player_B_4_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_4_                     =  self.random_input_inner_for_player_B_4 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_4.shape[0]: ] * self.beta


    def train_for_random_input_inner_player_B_5_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_5_                     =  self.random_input_inner_for_player_B_5 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_5.shape[0]: ] * self.beta


    def train_for_random_input_inner_player_B_6_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_6_                     =  self.random_input_inner_for_player_B_6 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_6.shape[0]: ] * self.beta


    def train_for_random_input_inner_player_B_7_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_7_                     =  self.random_input_inner_for_player_B_7 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_7.shape[0]: ] * self.beta



    def train_for_random_input_inner_player_B_8_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_8_                     =  self.random_input_inner_for_player_B_8 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_8.shape[0]: ] * self.beta



    def train_for_random_input_inner_player_B_9_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_9_                     =  self.random_input_inner_for_player_B_9 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_9.shape[0]: ] * self.beta



    def train_for_random_input_inner_player_B_10_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_10_                     =  self.random_input_inner_for_player_B_10 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_10.shape[0]: ] * self.beta



    def train_for_random_input_inner_player_B_11_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_11_                     =  self.random_input_inner_for_player_B_11 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_11.shape[0]: ] * self.beta



    def train_for_random_input_inner_player_B_12_(self,
                       layer_list,
                       output):

        layer_delta_list       = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               ) * np.array([self.life_signal_receiver_output_to_derivative(layer_list[-1])])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 1):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) ) * np.array([self.sigmoid_output_to_derivative(layer_list[- 1 - 1 - i] )])

            layer_delta_list.append(layer_delta)

        layer_delta_list = copy.deepcopy(layer_delta_list)

        self.random_input_inner_for_player_B_12_                     =  self.random_input_inner_for_player_B_12 + layer_delta_list[-1][0][-self.random_input_inner_for_player_B_12.shape[0]: ] * self.beta




    def deduct_from(self, random_input_inner_for_player_A,                                  #
                          random_input_inner_for_player_B,                                  #
                          ):

        size_of_pain_neurons             = np.true_divide(np.int(self.dims[-1]), 2)
        size_of_pain_neurons             = np.int(size_of_pain_neurons)

        self.random_input_inner_for_player_A   = random_input_inner_for_player_A                                                                                         #
        self.random_input_inner_for_player_B   = random_input_inner_for_player_B                                                                                         #

        strategy_neurons_size                  = random_input_inner_for_player_B.shape[0]

        for i in range(self.rounds):

            # ---------------------A-----------------------

            self.random_input_inner_for_player_A_2 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_3 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_4 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_5 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_6 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_7 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_8 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_9 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_10 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_11 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_A_12 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #

            self.momentum_A                        = np.zeros(strategy_neurons_size)
            self.momentum_A_2                      = np.zeros(strategy_neurons_size)
            self.momentum_A_3                      = np.zeros(strategy_neurons_size)
            self.momentum_A_4                      = np.zeros(strategy_neurons_size)
            self.momentum_A_5                      = np.zeros(strategy_neurons_size)
            self.momentum_A_6                      = np.zeros(strategy_neurons_size)
            self.momentum_A_7                      = np.zeros(strategy_neurons_size)
            self.momentum_A_8                      = np.zeros(strategy_neurons_size)
            self.momentum_A_9                      = np.zeros(strategy_neurons_size)
            self.momentum_A_10                      = np.zeros(strategy_neurons_size)
            self.momentum_A_11                      = np.zeros(strategy_neurons_size)
            self.momentum_A_12                      = np.zeros(strategy_neurons_size)


            #---------------------A SWARM-----------------------

            for j in range(self.rounds_inside_swarm):

                self.random_input_for_player_A       = self.processor(self.random_input_inner_for_player_A)                                                                  #
                self.random_input_for_player_B       = self.processor(self.random_input_inner_for_player_B)                                                                  #

                self.random_input_for_player_A_2     = self.processor(self.random_input_inner_for_player_A_2)  #
                self.random_input_for_player_A_3     = self.processor(self.random_input_inner_for_player_A_3)  #
                self.random_input_for_player_A_4     = self.processor(self.random_input_inner_for_player_A_4)  #
                self.random_input_for_player_A_5     = self.processor(self.random_input_inner_for_player_A_5)  #
                self.random_input_for_player_A_6     = self.processor(self.random_input_inner_for_player_A_6)  #
                self.random_input_for_player_A_7     = self.processor(self.random_input_inner_for_player_A_7)  #
                self.random_input_for_player_A_8     = self.processor(self.random_input_inner_for_player_A_8)  #
                self.random_input_for_player_A_9     = self.processor(self.random_input_inner_for_player_A_9)  #
                self.random_input_for_player_A_10     = self.processor(self.random_input_inner_for_player_A_10)  #
                self.random_input_for_player_A_11     = self.processor(self.random_input_inner_for_player_A_11)  #
                self.random_input_for_player_A_12     = self.processor(self.random_input_inner_for_player_A_12)  #


                layer_list              = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B  )))        #
                layer_list_2            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_2,  self.random_input_for_player_B  )))        #
                layer_list_3            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_3,  self.random_input_for_player_B  )))        #
                layer_list_4            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_4,  self.random_input_for_player_B  )))        #
                layer_list_5            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_5,  self.random_input_for_player_B  )))        #
                layer_list_6            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_6,  self.random_input_for_player_B  )))        #
                layer_list_7            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_7,  self.random_input_for_player_B  )))        #
                layer_list_8            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_8,  self.random_input_for_player_B  )))        #
                layer_list_9            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_9,  self.random_input_for_player_B  )))        #
                layer_list_10            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_10,  self.random_input_for_player_B  )))        #
                layer_list_11            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_11,  self.random_input_for_player_B  )))        #
                layer_list_12            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_12,  self.random_input_for_player_B  )))        #



                desired_output_for_player_A = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A[i]                        = 1
                    desired_output_for_player_A[i + size_of_pain_neurons] = layer_list[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_2 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_2[i]                        = 1
                    desired_output_for_player_A_2[i + size_of_pain_neurons] = layer_list_2[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_3 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_3[i]                        = 1
                    desired_output_for_player_A_3[i + size_of_pain_neurons] = layer_list_3[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_4 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_4[i]                        = 1
                    desired_output_for_player_A_4[i + size_of_pain_neurons] = layer_list_4[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_5 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_5[i]                        = 1
                    desired_output_for_player_A_5[i + size_of_pain_neurons] = layer_list_5[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_6 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_6[i] = 1
                    desired_output_for_player_A_6[i + size_of_pain_neurons] = layer_list_6[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_7 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_7[i]                        = 1
                    desired_output_for_player_A_7[i + size_of_pain_neurons] = layer_list_7[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_8 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_8[i]                        = 1
                    desired_output_for_player_A_8[i + size_of_pain_neurons] = layer_list_8[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_9 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_9[i]                        = 1
                    desired_output_for_player_A_9[i + size_of_pain_neurons] = layer_list_9[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_10 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_10[i]                        = 1
                    desired_output_for_player_A_10[i + size_of_pain_neurons] = layer_list_10[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_11 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_11[i]                        = 1
                    desired_output_for_player_A_11[i + size_of_pain_neurons] = layer_list_11[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_12 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_12[i]                        = 1
                    desired_output_for_player_A_12[i + size_of_pain_neurons] = layer_list_12[-1][i + size_of_pain_neurons]

                self.train_for_random_input_inner_player_A_(
                                     layer_list,
                                     desired_output_for_player_A)
                self.train_for_random_input_inner_player_A_2_(                                                                                                               #
                                     layer_list_2,                                                                                                                           #
                                     desired_output_for_player_A_2)                                                                                                            #
                self.train_for_random_input_inner_player_A_3_(                                                                                                               #
                                     layer_list_3,                                                                                                                           #
                                     desired_output_for_player_A_3)                                                                                                            #
                self.train_for_random_input_inner_player_A_4_(                                                                                                               #
                                     layer_list_4,                                                                                                                           #
                                     desired_output_for_player_A_4)                                                                                                            #
                self.train_for_random_input_inner_player_A_5_(                                                                                                               #
                                     layer_list_5,                                                                                                                           #
                                     desired_output_for_player_A_5)                                                                                                            #
                self.train_for_random_input_inner_player_A_6_(                                                                                                               #
                                     layer_list_6,                                                                                                                           #
                                     desired_output_for_player_A_6)                                                                                                            #
                self.train_for_random_input_inner_player_A_7_(                                                                                                               #
                                     layer_list_7,                                                                                                                           #
                                     desired_output_for_player_A_7)                                                                                                            #
                self.train_for_random_input_inner_player_A_8_(                                                                                                               #
                                     layer_list_8,                                                                                                                           #
                                     desired_output_for_player_A_8)                                                                                                            #
                self.train_for_random_input_inner_player_A_9_(                                                                                                               #
                                     layer_list_9,                                                                                                                           #
                                     desired_output_for_player_A_9)                                                                                                            #
                self.train_for_random_input_inner_player_A_10_(                                                                                                               #
                                     layer_list_10,                                                                                                                           #
                                     desired_output_for_player_A_10)                                                                                                            #
                self.train_for_random_input_inner_player_A_11_(                                                                                                               #
                                     layer_list_11,                                                                                                                           #
                                     desired_output_for_player_A_11)                                                                                                            #
                self.train_for_random_input_inner_player_A_12_(                                                                                                               #
                                     layer_list_12,                                                                                                                           #
                                     desired_output_for_player_A_12)                                                                                                            #








                layer_list              = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_  ),  self.random_input_for_player_B  )))       #
                layer_list_2            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_2_),  self.random_input_for_player_B  )))       #
                layer_list_3            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_3_),  self.random_input_for_player_B  )))       #
                layer_list_4            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_4_),  self.random_input_for_player_B  )))       #
                layer_list_5            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_5_),  self.random_input_for_player_B  )))       #
                layer_list_6            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_6_),  self.random_input_for_player_B  )))       #
                layer_list_7            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_7_),  self.random_input_for_player_B  )))       #
                layer_list_8            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_8_),  self.random_input_for_player_B  )))       #
                layer_list_9            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_9_),  self.random_input_for_player_B  )))       #
                layer_list_10            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_10_),  self.random_input_for_player_B  )))       #
                layer_list_11            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_11_),  self.random_input_for_player_B  )))       #
                layer_list_12            = self.generate_values_for_each_layer(np.concatenate(( self.sigmoid(self.random_input_inner_for_player_A_12_),  self.random_input_for_player_B  )))       #

                candidate = np.argmin(np.array([np.sum((np.ones(size_of_pain_neurons) - layer_list[-1][0:size_of_pain_neurons]) ** 2) ,                                      #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_2[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_3[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_4[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_5[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_6[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_7[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_8[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_9[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_10[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_11[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_12[-1][0:size_of_pain_neurons]) ** 2),                                     #
                                                ]))
                if candidate == 0:
                    self.chosen_A = self.random_input_inner_for_player_A_                                                                                                     #
                if candidate == 1:
                    self.chosen_A = self.random_input_inner_for_player_A_2_                                                                                                   #
                if candidate == 2:
                    self.chosen_A = self.random_input_inner_for_player_A_3_                                                                                                   #
                if candidate == 3:
                    self.chosen_A = self.random_input_inner_for_player_A_4_                                                                                                   #
                if candidate == 4:
                    self.chosen_A = self.random_input_inner_for_player_A_5_                                                                                                   #
                if candidate == 5:
                    self.chosen_A = self.random_input_inner_for_player_A_6_                                                                                                   #
                if candidate == 6:
                    self.chosen_A = self.random_input_inner_for_player_A_7_                                                                                                   #
                if candidate == 7:
                    self.chosen_A = self.random_input_inner_for_player_A_8_                                                                                                   #
                if candidate == 8:
                    self.chosen_A = self.random_input_inner_for_player_A_9_                                                                                                   #
                if candidate == 9:
                    self.chosen_A = self.random_input_inner_for_player_A_10_                                                                                                   #
                if candidate == 10:
                    self.chosen_A = self.random_input_inner_for_player_A_11_                                                                                                   #
                if candidate == 11:
                    self.chosen_A = self.random_input_inner_for_player_A_12_                                                                                                   #












                layer_list              = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B  )))        #
                layer_list_2            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_2,  self.random_input_for_player_B  )))        #
                layer_list_3            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_3,  self.random_input_for_player_B  )))        #
                layer_list_4            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_4,  self.random_input_for_player_B  )))        #
                layer_list_5            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_5,  self.random_input_for_player_B  )))        #
                layer_list_6            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_6,  self.random_input_for_player_B  )))        #
                layer_list_7            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_7,  self.random_input_for_player_B  )))        #
                layer_list_8            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_8,  self.random_input_for_player_B  )))        #
                layer_list_9            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_9,  self.random_input_for_player_B  )))        #
                layer_list_10            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_10,  self.random_input_for_player_B  )))        #
                layer_list_11            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_11,  self.random_input_for_player_B  )))        #
                layer_list_12            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A_12,  self.random_input_for_player_B  )))        #


                desired_output_for_player_A = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A[i]                        = 1
                    desired_output_for_player_A[i + size_of_pain_neurons] = layer_list[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_2 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_2[i]                        = 1
                    desired_output_for_player_A_2[i + size_of_pain_neurons] = layer_list_2[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_3 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_3[i]                        = 1
                    desired_output_for_player_A_3[i + size_of_pain_neurons] = layer_list_3[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_4 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_4[i]                        = 1
                    desired_output_for_player_A_4[i + size_of_pain_neurons] = layer_list_4[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_5 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_5[i]                        = 1
                    desired_output_for_player_A_5[i + size_of_pain_neurons] = layer_list_5[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_6 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_6[i] = 1
                    desired_output_for_player_A_6[i + size_of_pain_neurons] = layer_list_6[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_7 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_7[i]                        = 1
                    desired_output_for_player_A_7[i + size_of_pain_neurons] = layer_list_7[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_8 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_8[i]                        = 1
                    desired_output_for_player_A_8[i + size_of_pain_neurons] = layer_list_8[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_9 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_9[i]                        = 1
                    desired_output_for_player_A_9[i + size_of_pain_neurons] = layer_list_9[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_10 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_10[i]                        = 1
                    desired_output_for_player_A_10[i + size_of_pain_neurons] = layer_list_10[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_11 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_11[i]                        = 1
                    desired_output_for_player_A_11[i + size_of_pain_neurons] = layer_list_11[-1][i + size_of_pain_neurons]

                desired_output_for_player_A_12 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_A_12[i]                        = 1
                    desired_output_for_player_A_12[i + size_of_pain_neurons] = layer_list_12[-1][i + size_of_pain_neurons]

                self.train_for_random_input_inner_player_A(
                                     layer_list,
                                     desired_output_for_player_A)

                self.train_for_random_input_inner_player_A_2(                                                                                                                #
                                     layer_list_2,                                                                                                                           #
                                     desired_output_for_player_A_2)                                                                                                            #

                self.train_for_random_input_inner_player_A_3(                                                                                                                #
                                     layer_list_3,                                                                                                                           #
                                     desired_output_for_player_A_3)                                                                                                            #

                self.train_for_random_input_inner_player_A_4(                                                                                                                #
                                     layer_list_4,                                                                                                                           #
                                     desired_output_for_player_A_4)                                                                                                            #

                self.train_for_random_input_inner_player_A_5(                                                                                                                #
                                     layer_list_5,                                                                                                                           #
                                     desired_output_for_player_A_5)                                                                                                            #

                self.train_for_random_input_inner_player_A_6(                                                                                                                #
                                     layer_list_6,                                                                                                                           #
                                     desired_output_for_player_A_6)                                                                                                            #

                self.train_for_random_input_inner_player_A_7(                                                                                                                #
                                     layer_list_7,                                                                                                                           #
                                     desired_output_for_player_A_7)                                                                                                            #

                self.train_for_random_input_inner_player_A_8(                                                                                                                #
                                     layer_list_8,                                                                                                                           #
                                     desired_output_for_player_A_8)                                                                                                            #

                self.train_for_random_input_inner_player_A_9(                                                                                                                #
                                     layer_list_9,                                                                                                                           #
                                     desired_output_for_player_A_9)                                                                                                            #

                self.train_for_random_input_inner_player_A_10(                                                                                                                #
                                     layer_list_10,                                                                                                                           #
                                     desired_output_for_player_A_10)                                                                                                            #

                self.train_for_random_input_inner_player_A_11(                                                                                                                #
                                     layer_list_11,                                                                                                                           #
                                     desired_output_for_player_A_11)                                                                                                            #

                self.train_for_random_input_inner_player_A_12(                                                                                                                #
                                     layer_list_12,                                                                                                                           #
                                     desired_output_for_player_A_12)                                                                                                            #


            if candidate == 0:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A                                                                                                     #
            if candidate == 1:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_2                                                                                                   #
            if candidate == 2:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_3                                                                                                   #
            if candidate == 3:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_4                                                                                                   #
            if candidate == 4:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_5                                                                                                   #
            if candidate == 5:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_6                                                                                                   #
            if candidate == 6:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_7                                                                                                   #
            if candidate == 7:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_8                                                                                                   #
            if candidate == 8:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_9                                                                                                   #
            if candidate == 9:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_10                                                                                                   #
            if candidate == 10:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_11                                                                                                   #
            if candidate == 11:
                self.random_input_inner_for_player_A = self.random_input_inner_for_player_A_12                                                                                                   #

            # ---------------------B-----------------------
            self.random_input_inner_for_player_B_2 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_3 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_4 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_5 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_6 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_7 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_8 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_9 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_10 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_11 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #
            self.random_input_inner_for_player_B_12 = (np.random.random(strategy_neurons_size) - 0.5) * 5 - 0  #


            self.momentum_B                        = np.zeros(strategy_neurons_size)
            self.momentum_B_2                      = np.zeros(strategy_neurons_size)
            self.momentum_B_3                      = np.zeros(strategy_neurons_size)
            self.momentum_B_4                      = np.zeros(strategy_neurons_size)
            self.momentum_B_5                      = np.zeros(strategy_neurons_size)
            self.momentum_B_6                      = np.zeros(strategy_neurons_size)
            self.momentum_B_7                      = np.zeros(strategy_neurons_size)
            self.momentum_B_8                      = np.zeros(strategy_neurons_size)
            self.momentum_B_9                      = np.zeros(strategy_neurons_size)
            self.momentum_B_10                      = np.zeros(strategy_neurons_size)
            self.momentum_B_11                      = np.zeros(strategy_neurons_size)
            self.momentum_B_12                      = np.zeros(strategy_neurons_size)


            #---------------------B SWARM-----------------------

            for j in range(self.rounds_inside_swarm):

                self.random_input_for_player_A       = self.processor(self.random_input_inner_for_player_A)                                                                  #
                self.random_input_for_player_B       = self.processor(self.random_input_inner_for_player_B)                                                                  #

                self.random_input_for_player_B_2     = self.processor(self.random_input_inner_for_player_B_2)  #
                self.random_input_for_player_B_3     = self.processor(self.random_input_inner_for_player_B_3)  #
                self.random_input_for_player_B_4     = self.processor(self.random_input_inner_for_player_B_4)  #
                self.random_input_for_player_B_5     = self.processor(self.random_input_inner_for_player_B_5)  #
                self.random_input_for_player_B_6     = self.processor(self.random_input_inner_for_player_B_6)  #
                self.random_input_for_player_B_7     = self.processor(self.random_input_inner_for_player_B_7)  #
                self.random_input_for_player_B_8     = self.processor(self.random_input_inner_for_player_B_8)  #
                self.random_input_for_player_B_9     = self.processor(self.random_input_inner_for_player_B_9)  #
                self.random_input_for_player_B_10     = self.processor(self.random_input_inner_for_player_B_10)  #
                self.random_input_for_player_B_11     = self.processor(self.random_input_inner_for_player_B_11)  #
                self.random_input_for_player_B_12     = self.processor(self.random_input_inner_for_player_B_12)  #

                layer_list              = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B  )))        #
                layer_list_2            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_2)))        #
                layer_list_3            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_3)))        #
                layer_list_4            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_4)))        #
                layer_list_5            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_5)))        #
                layer_list_6            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_6)))        #
                layer_list_7            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_7)))        #
                layer_list_8            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_8)))        #
                layer_list_9            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_9)))        #
                layer_list_10            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_10)))        #
                layer_list_11            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_11)))        #
                layer_list_12            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_12)))        #

                desired_output_for_player_B = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B[i]                        = layer_list[-1][i]

                desired_output_for_player_B_2 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_2[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_2[i]                        = layer_list_2[-1][i]

                desired_output_for_player_B_3 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_3[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_3[i]                        = layer_list_3[-1][i]

                desired_output_for_player_B_4 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_4[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_4[i]                        = layer_list_4[-1][i]

                desired_output_for_player_B_5 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_5[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_5[i]                        = layer_list_5[-1][i]

                desired_output_for_player_B_6 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_6[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_6[i]                        = layer_list_6[-1][i]

                desired_output_for_player_B_7 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_7[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_7[i]                        = layer_list_7[-1][i]

                desired_output_for_player_B_8 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_8[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_8[i]                        = layer_list_8[-1][i]

                desired_output_for_player_B_9 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_9[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_9[i]                        = layer_list_9[-1][i]

                desired_output_for_player_B_10 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_10[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_10[i]                        = layer_list_10[-1][i]

                desired_output_for_player_B_11 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_11[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_11[i]                        = layer_list_11[-1][i]

                desired_output_for_player_B_12 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_12[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_12[i]                        = layer_list_12[-1][i]


                self.train_for_random_input_inner_player_B_(
                                     layer_list,
                                     desired_output_for_player_B)

                self.train_for_random_input_inner_player_B_2_(                                                                                                                #
                                     layer_list_2,                                                                                                                           #
                                     desired_output_for_player_B_2)                                                                                                            #

                self.train_for_random_input_inner_player_B_3_(                                                                                                                #
                                     layer_list_3,                                                                                                                           #
                                     desired_output_for_player_B_3)                                                                                                            #

                self.train_for_random_input_inner_player_B_4_(                                                                                                                #
                                     layer_list_4,                                                                                                                           #
                                     desired_output_for_player_B_4)                                                                                                            #

                self.train_for_random_input_inner_player_B_5_(                                                                                                                #
                                     layer_list_5,                                                                                                                           #
                                     desired_output_for_player_B_5)                                                                                                            #

                self.train_for_random_input_inner_player_B_6_(                                                                                                                #
                                     layer_list_6,                                                                                                                           #
                                     desired_output_for_player_B_6)                                                                                                            #

                self.train_for_random_input_inner_player_B_7_(                                                                                                                #
                                     layer_list_7,                                                                                                                           #
                                     desired_output_for_player_B_7)                                                                                                            #

                self.train_for_random_input_inner_player_B_8_(                                                                                                                #
                                     layer_list_8,                                                                                                                           #
                                     desired_output_for_player_B_8)                                                                                                            #

                self.train_for_random_input_inner_player_B_9_(                                                                                                                #
                                     layer_list_9,                                                                                                                           #
                                     desired_output_for_player_B_9)                                                                                                            #

                self.train_for_random_input_inner_player_B_10_(                                                                                                                #
                                     layer_list_10,                                                                                                                           #
                                     desired_output_for_player_B_10)                                                                                                            #

                self.train_for_random_input_inner_player_B_11_(                                                                                                                #
                                     layer_list_11,                                                                                                                           #
                                     desired_output_for_player_B_11)                                                                                                            #

                self.train_for_random_input_inner_player_B_12_(                                                                                                                #
                                     layer_list_12,                                                                                                                           #
                                     desired_output_for_player_B_12)                                                                                                            #








                layer_list              = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_  ))))       #
                layer_list_2            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_2_))))       #
                layer_list_3            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_3_))))       #
                layer_list_4            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_4_))))       #
                layer_list_5            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_5_))))       #
                layer_list_6            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_6_))))       #
                layer_list_7            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_7_))))       #
                layer_list_8            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_8_))))       #
                layer_list_9            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_9_))))       #
                layer_list_10            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_10_))))       #
                layer_list_11            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_11_))))       #
                layer_list_12            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.sigmoid(self.random_input_inner_for_player_B_12_))))       #

                candidate = np.argmin(np.array([np.sum((np.ones(size_of_pain_neurons) - layer_list[-1][-size_of_pain_neurons:]) ** 2) ,                                      #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_2[-1][-size_of_pain_neurons:]) ** 2) ,                                    #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_3[-1][-size_of_pain_neurons:]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_4[-1][-size_of_pain_neurons:]) ** 2),                                     #
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_5[-1][-size_of_pain_neurons:]) ** 2),
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_6[-1][-size_of_pain_neurons:]) ** 2),
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_7[-1][-size_of_pain_neurons:]) ** 2),
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_8[-1][-size_of_pain_neurons:]) ** 2),
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_9[-1][-size_of_pain_neurons:]) ** 2),
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_10[-1][-size_of_pain_neurons:]) ** 2),
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_11[-1][-size_of_pain_neurons:]) ** 2),
                                                np.sum((np.ones(size_of_pain_neurons) - layer_list_12[-1][-size_of_pain_neurons:]) ** 2),
                                                ]))
                if candidate == 0:
                    self.chosen_B = self.random_input_inner_for_player_B_                                                                                                     #
                if candidate == 1:
                    self.chosen_B = self.random_input_inner_for_player_B_2_                                                                                                   #
                if candidate == 2:
                    self.chosen_B = self.random_input_inner_for_player_B_3_                                                                                                   #
                if candidate == 3:
                    self.chosen_B = self.random_input_inner_for_player_B_4_                                                                                                   #
                if candidate == 4:
                    self.chosen_B = self.random_input_inner_for_player_B_5_                                                                                                   #
                if candidate == 5:
                    self.chosen_B = self.random_input_inner_for_player_B_6_                                                                                                   #
                if candidate == 6:
                    self.chosen_B = self.random_input_inner_for_player_B_7_                                                                                                   #
                if candidate == 7:
                    self.chosen_B = self.random_input_inner_for_player_B_8_                                                                                                   #
                if candidate == 8:
                    self.chosen_B = self.random_input_inner_for_player_B_9_                                                                                                   #
                if candidate == 9:
                    self.chosen_B = self.random_input_inner_for_player_B_10_                                                                                                   #
                if candidate == 10:
                    self.chosen_B = self.random_input_inner_for_player_B_11_                                                                                                   #
                if candidate == 11:
                    self.chosen_B = self.random_input_inner_for_player_B_12_                                                                                                   #









                layer_list              = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B  )))        #
                layer_list_2            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_2)))        #
                layer_list_3            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_3)))        #
                layer_list_4            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_4)))        #
                layer_list_5            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_5)))        #
                layer_list_6            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_6)))        #
                layer_list_7            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_7)))        #
                layer_list_8            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_8)))        #
                layer_list_9            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_9)))        #
                layer_list_10            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_10)))        #
                layer_list_11            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_11)))        #
                layer_list_12            = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A  ,  self.random_input_for_player_B_12)))        #

                desired_output_for_player_B = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B[i]                        = layer_list[-1][i]

                desired_output_for_player_B_2 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_2[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_2[i]                        = layer_list_2[-1][i]

                desired_output_for_player_B_3 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_3[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_3[i]                        = layer_list_3[-1][i]

                desired_output_for_player_B_4 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_4[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_4[i]                        = layer_list_4[-1][i]

                desired_output_for_player_B_5 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_5[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_5[i]                        = layer_list_5[-1][i]

                desired_output_for_player_B_6 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_6[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_6[i]                        = layer_list_6[-1][i]

                desired_output_for_player_B_7 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_7[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_7[i]                        = layer_list_7[-1][i]

                desired_output_for_player_B_8 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_8[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_8[i]                        = layer_list_8[-1][i]

                desired_output_for_player_B_9 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_9[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_9[i]                        = layer_list_9[-1][i]

                desired_output_for_player_B_10 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_10[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_10[i]                        = layer_list_10[-1][i]

                desired_output_for_player_B_11 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_11[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_11[i]                        = layer_list_11[-1][i]

                desired_output_for_player_B_12 = np.zeros(size_of_pain_neurons * 2)
                for i in range(size_of_pain_neurons):
                    desired_output_for_player_B_12[i + size_of_pain_neurons] = 1
                    desired_output_for_player_B_12[i]                        = layer_list_12[-1][i]

                self.train_for_random_input_inner_player_B(
                                     layer_list,
                                     desired_output_for_player_B)

                self.train_for_random_input_inner_player_B_2(                                                                                                                #
                                     layer_list_2,                                                                                                                           #
                                     desired_output_for_player_B_2)                                                                                                            #

                self.train_for_random_input_inner_player_B_3(                                                                                                                #
                                     layer_list_3,                                                                                                                           #
                                     desired_output_for_player_B_3)                                                                                                            #

                self.train_for_random_input_inner_player_B_4(                                                                                                                #
                                     layer_list_4,                                                                                                                           #
                                     desired_output_for_player_B_4)                                                                                                            #

                self.train_for_random_input_inner_player_B_5(                                                                                                                #
                                     layer_list_5,                                                                                                                           #
                                     desired_output_for_player_B_5)                                                                                                            #

                self.train_for_random_input_inner_player_B_6(                                                                                                                #
                                     layer_list_6,                                                                                                                           #
                                     desired_output_for_player_B_6)                                                                                                            #

                self.train_for_random_input_inner_player_B_7(                                                                                                                #
                                     layer_list_7,                                                                                                                           #
                                     desired_output_for_player_B_7)                                                                                                            #

                self.train_for_random_input_inner_player_B_8(                                                                                                                #
                                     layer_list_8,                                                                                                                           #
                                     desired_output_for_player_B_8)                                                                                                            #

                self.train_for_random_input_inner_player_B_9(                                                                                                                #
                                     layer_list_9,                                                                                                                           #
                                     desired_output_for_player_B_9)                                                                                                            #

                self.train_for_random_input_inner_player_B_10(                                                                                                                #
                                     layer_list_10,                                                                                                                           #
                                     desired_output_for_player_B_10)                                                                                                            #

                self.train_for_random_input_inner_player_B_11(                                                                                                                #
                                     layer_list_11,                                                                                                                           #
                                     desired_output_for_player_B_11)                                                                                                            #

                self.train_for_random_input_inner_player_B_12(                                                                                                                #
                                     layer_list_12,                                                                                                                           #
                                     desired_output_for_player_B_12)                                                                                                            #

            if candidate == 0:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B                                                                                                     #
            if candidate == 1:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_2                                                                                                   #
            if candidate == 2:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_3                                                                                                   #
            if candidate == 3:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_4                                                                                                   #
            if candidate == 4:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_5                                                                                                   #
            if candidate == 5:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_6                                                                                                   #
            if candidate == 6:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_7                                                                                                   #
            if candidate == 7:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_8                                                                                                   #
            if candidate == 8:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_9                                                                                                   #
            if candidate == 9:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_10                                                                                                   #
            if candidate == 10:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_11                                                                                                   #
            if candidate == 11:
                self.random_input_inner_for_player_B = self.random_input_inner_for_player_B_12                                                                                                  #

        return self.random_input_inner_for_player_A, self.random_input_inner_for_player_B






