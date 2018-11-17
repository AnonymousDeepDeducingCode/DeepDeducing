import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
np.random.seed(0)

class Basic_Research_Model(object):
    def __init__(self, input_dim, output_dim, alpha, epochs, beta, rounds, function):

        self.layer_0_dim                  = input_dim
        self.layer_1_dim                  = 30
        self.layer_2_dim                  = output_dim

        self.alpha                        = alpha
        self.epochs                       = epochs
        self.beta                         = beta
        self.rounds                       = rounds


        self.function                     = function

        self.synapse_layer_0_to_layer_1   ,\
        self.synapse_layer_1_to_layer_2   = self.initialize_weights()

        self.synapse_layer_0_to_layer_1_update          = np.zeros_like( self.synapse_layer_0_to_layer_1 )
        self.synapse_layer_1_to_layer_2_update          = np.zeros_like( self.synapse_layer_1_to_layer_2 )

    def initialize_weights(self):

        synapse_layer_0_to_layer_1                                              = (np.random.random((self.layer_0_dim                                 , self.layer_1_dim                           )) -0.5 ) * 15
        synapse_layer_1_to_layer_2                                              = (np.random.random((self.layer_1_dim                                 , self.layer_2_dim                           )) -0.5 ) * 15

        return  synapse_layer_0_to_layer_1,\
                synapse_layer_1_to_layer_2

    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output

    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

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
        return output

    def generate_values_for_each_layer(self, input):

        layer_0                   = copy.deepcopy(np.array(input))

        layer_1                   = self.sigmoid(np.dot(layer_0                          , self.synapse_layer_0_to_layer_1                                                          ) )

        layer_2                   = self.sigmoid(np.dot(layer_1                          , self.synapse_layer_1_to_layer_2                                                          ) )

        return   layer_0             ,\
                    layer_1             ,\
                    layer_2


    def train_for_each(self,
                       layer_0,
                       layer_1,
                       layer_2,
                       output):

        layer_2_error                           = np.array([output - layer_2])

        layer_2_delta                           = (layer_2_error                                                                                            ) * np.array([self.sigmoid_output_to_derivative(layer_2)])

        layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_1)])

        self.synapse_layer_0_to_layer_1_update                            += np.atleast_2d(layer_0                                                          ).T.dot(layer_1_delta                              )
        self.synapse_layer_1_to_layer_2_update                            += np.atleast_2d(layer_1                                                          ).T.dot(layer_2_delta                              )

        self.synapse_layer_0_to_layer_1                                   += self.synapse_layer_0_to_layer_1_update                   * self.alpha
        self.synapse_layer_1_to_layer_2                                   += self.synapse_layer_1_to_layer_2_update                   * self.alpha

        self.synapse_layer_0_to_layer_1_update                            *= 0
        self.synapse_layer_1_to_layer_2_update                            *= 0



    def fit(self, input_list, output_list):

        for i in range(self.epochs):

            random_index = np.random.randint(input_list.shape[0])
            input        = input_list[random_index]
            output       = output_list[random_index]

            layer_0,\
            layer_1, \
            layer_2  = self.generate_values_for_each_layer(input)

            self.train_for_each(
                       layer_0,
                       layer_1,
                       layer_2,
                       output)


        return self



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



    def train_for_random_input_inner_player_A(self,
                       layer_0,
                       layer_1,
                       layer_2,
                       output):

        layer_2_error                           = np.array([output - layer_2])

        layer_2_delta                           = (layer_2_error                                                                                            ) * np.array([self.sigmoid_output_to_derivative(layer_2)])

        layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_1)])

        layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * np.array([self.processor_output_to_derivative(layer_0)])


        self.random_input_inner_for_player_A += layer_0_delta[0][0:self.random_input_inner_for_player_A.shape[0]] * self.beta




    def train_for_random_input_inner_player_B(self,
                       layer_0,
                       layer_1,
                       layer_2,
                       output):

        layer_2_error                           = np.array([output - layer_2])

        layer_2_delta                           = (layer_2_error                                                                                            ) * np.array([self.sigmoid_output_to_derivative(layer_2)])

        layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_1)])

        layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * np.array([self.processor_output_to_derivative(layer_0)])


        self.random_input_inner_for_player_B += layer_0_delta[0][-self.random_input_inner_for_player_B.shape[0]: ] * self.beta




    def deduct_from(self, random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B, axarr, color_middle, marker_middle):

        self.random_input_inner_for_player_A = random_input_inner_for_player_A
        self.random_input_inner_for_player_B = random_input_inner_for_player_B


        for i in range(self.rounds):

            #---------------------A-----------------------

            self.random_input_for_player_A       = self.processor(self.random_input_inner_for_player_A)
            self.random_input_for_player_B       = self.processor(self.random_input_inner_for_player_B)

            layer_0,\
            layer_1, \
            layer_2 = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A,  self.random_input_for_player_B)))

            desired_output_for_player_A    = copy.deepcopy(layer_2)
            for j in range(10):
                desired_output_for_player_A[j] = desired_payoff_for_player_A


            self.train_for_random_input_inner_player_A(
                                 layer_0,
                                 layer_1,
                                 layer_2,
                                 desired_output_for_player_A)

            #---------------------B-----------------------

            self.random_input_for_player_A       = self.processor(self.random_input_inner_for_player_A)
            self.random_input_for_player_B       = self.processor(self.random_input_inner_for_player_B)


            layer_0,\
            layer_1, \
            layer_2  = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A,  self.random_input_for_player_B)))

            desired_output_for_player_B    = copy.deepcopy(layer_2)
            for j in range(10):
                desired_output_for_player_B[10+j] = desired_payoff_for_player_B


            self.train_for_random_input_inner_player_B(
                                 layer_0,
                                 layer_1,
                                 layer_2,
                                 desired_output_for_player_B)


            if i % 25 == 0:

                axarr.plot(np.round(self.processor(self.random_input_inner_for_player_B), 2)[0],np.round(self.processor(self.random_input_inner_for_player_A), 2)[0], markersize = 1.5, c=  color_middle, marker = marker_middle)


        return self.random_input_inner_for_player_A, self.random_input_inner_for_player_B

