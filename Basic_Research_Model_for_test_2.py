import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
np.random.seed(0)

class Basic_Research_Model(object):
    def __init__(self, dims, number_of_layers, alpha, epochs, beta, rounds, signal_receiver, signal_transfer, randomness, function, training_method, stachostic):

        self.dims                         = dims

        self.number_of_layers             = number_of_layers

        self.alpha                        = alpha
        self.epochs                       = epochs
        self.beta                         = beta
        self.rounds                       = rounds

        self.signal_receiver              = signal_receiver
        self.signal_transfer              = signal_transfer
        self.randomness                   = randomness
        self.function                     = function
        self.training_method              = training_method
        self.stachostic                   = stachostic

        self.synapse_list                 = self.initialize_weights()

        self.synapse_list_update          = np.zeros_like( self.synapse_list)

        strategy_neurons_size             = np.true_divide(np.int(dims[0]), 2)
        strategy_neurons_size             = np.int(strategy_neurons_size)
        self.momentum_A                   = np.zeros(strategy_neurons_size)
        self.momentum_B                   = np.zeros(strategy_neurons_size)

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



        if self.training_method == "traditional":
            if self.randomness == "True":
                self.random_input_inner_for_player_A[self.random_target] += layer_delta_list[-1][0][0:self.random_input_inner_for_player_A.shape[0]][self.random_target] * self.beta
            else:
                self.random_input_inner_for_player_A                     += layer_delta_list[-1][0][0:self.random_input_inner_for_player_A.shape[0]] * self.beta
        if self.training_method == "momentum":
            if self.randomness == "True":
                self.momentum_A[self.random_target]                       = self.momentum_A[self.random_target] * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A.shape[0]][self.random_target] * self.beta
                self.random_input_inner_for_player_A[self.random_target] += self.momentum_A[self.random_target]
            else:
                self.momentum_A                                           = self.momentum_A * self.beta + layer_delta_list[-1][0][0:self.random_input_inner_for_player_A.shape[0]] * self.beta
                self.random_input_inner_for_player_A                     += self.momentum_A



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



        if self.training_method == "traditional":
            if self.randomness == "True":
                self.random_input_inner_for_player_B[self.random_target] += layer_delta_list[-1][0][-self.random_input_inner_for_player_B.shape[0]: ][self.random_target] * self.beta
            else:
                self.random_input_inner_for_player_B                     += layer_delta_list[-1][0][-self.random_input_inner_for_player_B.shape[0]: ] * self.beta
        if self.training_method == "momentum":
            if self.randomness == "True":
                self.momentum_B[self.random_target]                       = self.momentum_B[self.random_target] * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B.shape[0]:][self.random_target] * self.beta
                self.random_input_inner_for_player_B[self.random_target] += self.momentum_B[self.random_target]
            else:
                self.momentum_B                                           = self.momentum_B * self.beta + layer_delta_list[-1][0][-self.random_input_inner_for_player_B.shape[0]:] * self.beta
                self.random_input_inner_for_player_B                     += self.momentum_B



    def deduct_from(self, random_input_inner_for_player_A, random_input_inner_for_player_B, size_of_pain_neurons):

        self.random_input_inner_for_player_A = random_input_inner_for_player_A
        self.random_input_inner_for_player_B = random_input_inner_for_player_B

        for i in range(self.rounds):

            #---------------------A-----------------------

            self.random_input_for_player_A       = self.processor(self.random_input_inner_for_player_A)
            self.random_input_for_player_B       = self.processor(self.random_input_inner_for_player_B)

            self.random_target      = np.random.randint(self.random_input_inner_for_player_A.shape[0])

            layer_list              = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A,  self.random_input_for_player_B)))

            desired_output_for_player_A = np.zeros(size_of_pain_neurons * 2)
            for i in range(size_of_pain_neurons):
                desired_output_for_player_A[i]                        = 1
                desired_output_for_player_A[i + size_of_pain_neurons] = layer_list[-1][i + size_of_pain_neurons]
                if self.stachostic == "True":
                    dropped_neuron                                    = np.random.randint(size_of_pain_neurons)
                    desired_output_for_player_A[dropped_neuron]       = 0

            self.train_for_random_input_inner_player_A(
                                 layer_list,
                                 desired_output_for_player_A)

            #---------------------B-----------------------

            self.random_input_for_player_A       = self.processor(self.random_input_inner_for_player_A)
            self.random_input_for_player_B       = self.processor(self.random_input_inner_for_player_B)

            self.random_target      = np.random.randint(self.random_input_inner_for_player_B.shape[0])

            layer_list              = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A,  self.random_input_for_player_B)))

            desired_output_for_player_B = np.zeros(size_of_pain_neurons * 2)
            for i in range(size_of_pain_neurons):
                desired_output_for_player_B[i + size_of_pain_neurons] = 1
                desired_output_for_player_B[i]                        = layer_list[-1][i]
                if self.stachostic == "True":
                    dropped_neuron                                                           = np.random.randint(size_of_pain_neurons)
                    desired_output_for_player_A[dropped_neuron + size_of_pain_neurons]       = 0

            self.train_for_random_input_inner_player_B(
                                 layer_list,
                                 desired_output_for_player_B)

        return self.random_input_inner_for_player_A, self.random_input_inner_for_player_B






