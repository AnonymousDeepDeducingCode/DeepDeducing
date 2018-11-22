import copy
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
np.random.seed(0)

class Game_Net_Problem_Solving(object):
    def __init__(self, input_dim, output_dim, alpha, epochs, beta, rounds):

        self.layer_0_dim                  = input_dim
        self.layer_1_dim                  = input_dim * 1
        self.layer_2_dim                  = input_dim * 1
        self.layer_3_dim                  = output_dim

        self.alpha                        = alpha
        self.epochs                       = epochs
        self.beta                         = beta
        self.rounds                       = rounds

        self.synapse_layer_0_to_layer_1   ,\
        self.synapse_layer_1_to_layer_2   ,\
        self.synapse_layer_2_to_layer_3   ,\
        self.conduit_layer_0_to_layer_3   = self.initialize_weights()

        self.synapse_layer_0_to_layer_1_update          = np.zeros_like( self.synapse_layer_0_to_layer_1 )
        self.synapse_layer_1_to_layer_2_update          = np.zeros_like( self.synapse_layer_1_to_layer_2 )
        self.synapse_layer_2_to_layer_3_update          = np.zeros_like( self.synapse_layer_2_to_layer_3 )
        self.conduit_layer_0_to_layer_3_update          = np.zeros_like( self.conduit_layer_0_to_layer_3 )

    def initialize_weights(self):

        synapse_layer_0_to_layer_1                                              = (np.random.random((self.layer_0_dim                                 , self.layer_1_dim                           )) -0.5 ) * 0.1
        synapse_layer_1_to_layer_2                                              = (np.random.random((self.layer_1_dim                                 , self.layer_2_dim                           )) -0.5 ) * 0.1
        synapse_layer_2_to_layer_3                                              = (np.random.random((self.layer_2_dim                                 , self.layer_3_dim                           )) -0.5 ) * 0.1
        conduit_layer_0_to_layer_3                                              = (np.ones(self.layer_3_dim))

        return  synapse_layer_0_to_layer_1,\
                synapse_layer_1_to_layer_2,\
                synapse_layer_2_to_layer_3,\
                conduit_layer_0_to_layer_3

    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output

    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

    def tanh(self, x):
        output = np.tanh(x)
        return output

    def tanh_output_to_derivative(self, output):
        output = 1 - output*output
        return output

    def generate_values_for_each_layer(self, X):

        layer_0                   = np.array(X)

        layer_1                   = self.sigmoid(np.dot(layer_0                          , self.synapse_layer_0_to_layer_1                                                          ) )

        layer_2                   = self.sigmoid(np.dot(layer_1                          , self.synapse_layer_1_to_layer_2                                                          ) )

        layer_3                   = layer_0[0:self.layer_3_dim]                          * self.conduit_layer_0_to_layer_3                                                            +\
                                    self.tanh(np.dot(layer_2                             , self.synapse_layer_2_to_layer_3                                                          ) )

        return   layer_0             ,\
                    layer_1             ,\
                    layer_2             ,\
                    layer_3

    def train_for_each(self,
                       layer_0,
                       layer_1,
                       layer_2,
                       layer_3,
                       Y):

        layer_3_error                           = np.array([Y - layer_3])

        layer_3_delta                           = (layer_3_error                                                                                            ) * np.array([self.tanh_output_to_derivative(np.dot(layer_2, self.synapse_layer_2_to_layer_3))])

        layer_2_delta                           = (layer_3_delta.dot(self.synapse_layer_2_to_layer_3.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_2)])

        layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_1)])

        self.synapse_layer_0_to_layer_1_update                            += np.atleast_2d(layer_0                                                          ).T.dot(layer_1_delta                              )
        self.synapse_layer_1_to_layer_2_update                            += np.atleast_2d(layer_1                                                          ).T.dot(layer_2_delta                              )
        self.synapse_layer_2_to_layer_3_update                            += np.atleast_2d(layer_2                                                          ).T.dot(layer_3_delta                              )

        self.synapse_layer_0_to_layer_1                                   += self.synapse_layer_0_to_layer_1_update                   * self.alpha
        self.synapse_layer_1_to_layer_2                                   += self.synapse_layer_1_to_layer_2_update                   * self.alpha
        self.synapse_layer_2_to_layer_3                                   += self.synapse_layer_2_to_layer_3_update                   * self.alpha

        self.synapse_layer_0_to_layer_1_update                            *= 0
        self.synapse_layer_1_to_layer_2_update                            *= 0
        self.synapse_layer_2_to_layer_3_update                            *= 0


    def fit(self, X, Y):

        layer_0,\
        layer_1, \
        layer_2, \
        layer_3  = self.generate_values_for_each_layer(X)

        self.train_for_each(
                   layer_0,
                   layer_1,
                   layer_2,
                   layer_3,
                   Y)

        return self

#=======================================================================================================================

    def DFNN_to_RNN(self, input_dim, output_dim):

        self.synapse_layer_3_to_future_layer_1         = self.synapse_layer_0_to_layer_1[0:output_dim       , :]
        self.conduit_layer_3_to_future_layer_3         = self.conduit_layer_0_to_layer_3
        self.synapse_layer_0_to_layer_1                = self.synapse_layer_0_to_layer_1[-input_dim:        , :]
        self.synapse_layer_1_to_layer_2                = self.synapse_layer_1_to_layer_2
        self.synapse_layer_2_to_layer_3                = self.synapse_layer_2_to_layer_3

        self.synapse_layer_3_to_future_layer_1_update  = np.zeros_like(self.synapse_layer_3_to_future_layer_1 )
        self.conduit_layer_3_to_future_layer_3_update  = np.zeros_like(self.conduit_layer_3_to_future_layer_3 )
        self.synapse_layer_0_to_layer_1_update         = np.zeros_like(self.synapse_layer_0_to_layer_1        )
        self.synapse_layer_1_to_layer_2_update         = np.zeros_like(self.synapse_layer_1_to_layer_2        )
        self.synapse_layer_2_to_layer_3_update         = np.zeros_like(self.synapse_layer_2_to_layer_3        )


    def generate_values_for_each_layer_RNN(self, initial_map, selected_movements):

        layer_0_values = list()
        layer_0_values.append(np.zeros(self.layer_0_dim - self.layer_3_dim))
        layer_0_values.append(np.zeros(self.layer_0_dim - self.layer_3_dim))
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.layer_1_dim))
        layer_1_values.append(np.zeros(self.layer_1_dim))
        layer_2_values = list()
        layer_2_values.append(np.zeros(self.layer_2_dim))
        layer_2_values.append(np.zeros(self.layer_2_dim))
        layer_3_values = list()
        layer_3_values.append(np.zeros_like(initial_map))
        layer_3_values.append(initial_map)

        for position in range(np.array(selected_movements).shape[0]):

            layer_0                   = selected_movements[position]

            layer_1                   = self.sigmoid(np.dot(layer_0                          , self.synapse_layer_0_to_layer_1                                                          ) +
                                                     np.dot(layer_3_values[-1]               , self.synapse_layer_3_to_future_layer_1                                                   ) )

            layer_2                   = self.sigmoid(np.dot(layer_1                          , self.synapse_layer_1_to_layer_2                                                          ) )

            layer_3                   = layer_3_values[-1]                                   * self.conduit_layer_3_to_future_layer_3                                                     +\
                                        self.tanh(np.dot(layer_2                             , self.synapse_layer_2_to_layer_3                                                          ) )

            layer_0_values                   .append(copy.deepcopy(    layer_0                      ))
            layer_1_values                   .append(copy.deepcopy(    layer_1                      ))
            layer_2_values                   .append(copy.deepcopy(    layer_2                      ))
            layer_3_values                   .append(copy.deepcopy(    layer_3                      ))

        layer_0_values                   = np.array(layer_0_values               )
        layer_1_values                   = np.array(layer_1_values               )
        layer_2_values                   = np.array(layer_2_values               )
        layer_3_values                   = np.array(layer_3_values               )

        return   layer_0_values             ,\
                    layer_1_values             ,\
                    layer_2_values             ,\
                    layer_3_values


    def train_for_inputs(self,selected_movements,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_3_values,
                       final_map):

        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_3_delta_2                      = np.array([np.zeros(self.layer_3_dim)])
        layer_3_error                               = np.array([final_map - layer_3_values[-1]])

        layer_0_delta_list                          = list()

        for position in range(np.array(selected_movements).shape[0]):

            layer_0                                 = np.array([layer_0_values[-position - 1]])
            layer_1                                 = np.array([layer_1_values[-position - 1]])
            layer_2                                 = np.array([layer_2_values[-position - 1]])
            layer_3                                 = np.array([layer_3_values[-position - 1]])

            layer_3_delta_1                        = (future_layer_1_delta.dot(self.synapse_layer_3_to_future_layer_1.T                                         ) +
                                                      future_layer_3_delta_2 * self.conduit_layer_3_to_future_layer_3                                             +
                                                      layer_3_error                                                                                             ) * self.tanh_output_to_derivative(np.dot(layer_2, self.synapse_layer_2_to_layer_3))

            layer_3_delta_2                        = (future_layer_1_delta.dot(self.synapse_layer_3_to_future_layer_1.T                                         ) +
                                                      future_layer_3_delta_2 * self.conduit_layer_3_to_future_layer_3                                             +
                                                      layer_3_error                                                                                             )

            layer_2_delta                           = (layer_3_delta_1.dot(self.synapse_layer_2_to_layer_3.T                                                  ) ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)

            layer_0_delta_list.append         (copy.deepcopy(layer_0_delta[0]))



            future_layer_1_delta                           = layer_1_delta
            future_layer_3_delta_2                         = layer_3_delta_2
            layer_3_error                                  = np.array([np.zeros(self.layer_3_dim)])

        self.selected_movements_inner[self.present_trainee] += layer_0_delta_list[-self.present_trainee-1] * self.beta       #----------------------------------------------




    def deduct_from(self, initial_map, selected_movements, final_map):

        self.selected_movements_inner = selected_movements

        for i in range(self.rounds):

            for j in range(self.selected_movements_inner.shape[0]):


                self.present_trainee = self.selected_movements_inner.shape[0] - j - 1




                selected_movements = self.sigmoid(self.selected_movements_inner)

                layer_0_values, \
                layer_1_values, \
                layer_2_values, \
                layer_3_values = self.generate_values_for_each_layer_RNN(initial_map, selected_movements)

                self.train_for_inputs(selected_movements,
                                      layer_0_values,
                                      layer_1_values,
                                      layer_2_values,
                                      layer_3_values,
                                      final_map)

        return self.selected_movements_inner
