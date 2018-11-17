import copy
import numpy as np

np.random.seed(0)

class Game_Net(object):
    def __init__(self, middle_dim, alpha , epochs , beta, rounds, strategies_so_far, steps_foreseen):


        self.layer_0_dim                  = middle_dim
        self.layer_1_dim                  = middle_dim + 22
        self.layer_2_dim                  = 2

        self.alpha                        = alpha
        self.epochs                       = epochs
        self.beta                         = beta
        self.rounds                       = rounds


        self.synapse_layer_0_to_layer_1                                         ,\
        self.synapse_layer_1_to_layer_2                                         ,\
        self.synapse_layer_1_to_future_layer_1                                  = self.initialize_weights()

        self.synapse_layer_0_to_layer_1_update                                  = np.zeros_like(self.synapse_layer_0_to_layer_1                          )
        self.synapse_layer_1_to_layer_2_update                                  = np.zeros_like(self.synapse_layer_1_to_layer_2                          )
        self.synapse_layer_1_to_future_layer_1_update                           = np.zeros_like(self.synapse_layer_1_to_future_layer_1                   )

        self.player_1_goal = np.array([1, 0])
        self.player_2_goal = np.array([0, 1])




        self.strategies_so_far            = strategies_so_far

        self.steps_foreseen               = steps_foreseen

        self.strategies_foreseen          = self.initialize_strategies_foreseen()


    def initialize_weights(self):

        synapse_layer_0_to_layer_1                                              = (np.random.random((self.layer_0_dim                                 , self.layer_1_dim                           )) -0.5 ) * 0.1
        synapse_layer_1_to_layer_2                                              = (np.random.random((self.layer_1_dim                                 , self.layer_2_dim                           )) -0.5 ) * 0.1
        synapse_layer_1_to_future_layer_1                                       = (np.random.random((self.layer_1_dim                                 , self.layer_1_dim                           )) -0.5 ) * 0.1


        return  synapse_layer_0_to_layer_1,\
                synapse_layer_1_to_layer_2, \
                synapse_layer_1_to_future_layer_1


    def initialize_strategies_foreseen(self):
        strategies_foreseen = list()
        for i in range(self.steps_foreseen):
            strategies_foreseen.append( (np.random.rand(self.strategies_so_far.shape[1]) - 0.5) * 0.1 )
        return np.array(strategies_foreseen)

    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output

    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

    def generate_values_for_each_layer(self, selected_sentence):
        layer_0_values = list()
        layer_0_values.append(np.zeros(self.layer_0_dim))
        layer_0_values.append(np.zeros(self.layer_0_dim))
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.layer_1_dim))
        layer_1_values.append(np.zeros(self.layer_1_dim))
        layer_2_values = list()
        layer_2_values.append(np.zeros(self.layer_2_dim))
        layer_2_values.append(np.zeros(self.layer_2_dim))

        for position in range(np.array(selected_sentence).shape[0]):

            layer_0                   = selected_sentence[position]

            layer_1                   = self.sigmoid(np.dot(layer_0                          , self.synapse_layer_0_to_layer_1                                                          ) +
                                                     np.dot(layer_1_values[-1]               , self.synapse_layer_1_to_future_layer_1                                                   ) )
            layer_2                   = self.sigmoid(np.dot(layer_1                          , self.synapse_layer_1_to_layer_2                                                          ) )


            layer_0_values                   .append(copy.deepcopy(    layer_0                      ))
            layer_1_values                   .append(copy.deepcopy(    layer_1                      ))
            layer_2_values                   .append(copy.deepcopy(    layer_2                      ))


        layer_0_values                   = np.array(layer_0_values               )
        layer_1_values                   = np.array(layer_1_values               )
        layer_2_values                   = np.array(layer_2_values               )

        return   layer_0_values             ,\
                    layer_1_values             ,\
                    layer_2_values



    def train_for_each(self, selected_sentence,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_2_opposite_value):

        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1]])

        for position in range(np.array(selected_sentence).shape[0]):

            layer_0                                 = np.array([layer_0_values[-position - 1]])
            layer_1                                 = np.array([layer_1_values[-position - 1]])
            layer_2                                 = np.array([layer_2_values[-position - 1]])



            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) ) * self.sigmoid_output_to_derivative(layer_1)



            self.synapse_layer_0_to_layer_1_update                        += np.atleast_2d(layer_0                                                                            ).T.dot(layer_1_delta                              )
            self.synapse_layer_1_to_layer_2_update                        += np.atleast_2d(layer_1                                                                            ).T.dot(layer_2_delta                              )
            self.synapse_layer_1_to_future_layer_1_update                 += np.atleast_2d(layer_1                                                                            ).T.dot(future_layer_1_delta                       )

            future_layer_1_delta                    = layer_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.synapse_layer_0_to_layer_1                                   += self.synapse_layer_0_to_layer_1_update                   * self.alpha
        self.synapse_layer_1_to_layer_2                                   += self.synapse_layer_1_to_layer_2_update                   * self.alpha
        self.synapse_layer_1_to_future_layer_1                            += self.synapse_layer_1_to_future_layer_1_update            * self.alpha


        self.synapse_layer_0_to_layer_1_update                            *= 0
        self.synapse_layer_1_to_layer_2_update                            *= 0
        self.synapse_layer_1_to_future_layer_1_update                     *= 0




    def fit(self, X, Y):
        for j in range(self.epochs):

            random_int        = np.random.randint(X.shape[0])
            selected_sentence = np.array( X[random_int ] )
            selected_result   = np.array( Y[random_int ] )

            layer_0_values,\
            layer_1_values, \
            layer_2_values = self.generate_values_for_each_layer(selected_sentence)

            self.train_for_each(selected_sentence,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       selected_result)

        return self



#==================================TRAINING BY BACK DEDUCTION==============================================



    def train_for_strategy_foreseen(self,selected_sentence,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_2_opposite_value,
                       step_foreseen):


        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]):

            layer_0                                 = np.array([layer_0_values[-position - 1]])
            layer_1                                 = np.array([layer_1_values[-position - 1]])
            layer_2                                 = np.array([layer_2_values[-position - 1]])




            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_1_delta                    = layer_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])


        self.strategies_foreseen[step_foreseen - 1] += layer_0_delta_list[- ( self.strategies_so_far.shape[0] + step_foreseen)] * self.beta





    def predict(self):


        for i in range(self.rounds):



            for j in range(self.steps_foreseen):





                layer_0_values,\
                layer_1_values, \
                layer_2_values = self.generate_values_for_each_layer( np.concatenate(( self.strategies_so_far, self.sigmoid( self.strategies_foreseen )  )) )




                if (self.strategies_so_far.shape[0] + self.steps_foreseen - j) % 2 == 1:

                    desired_result = self.player_1_goal

                else:

                    desired_result = self.player_2_goal



                self.train_for_strategy_foreseen(np.concatenate(( self.strategies_so_far, self.sigmoid( self.strategies_foreseen )  )),
                           layer_0_values,
                           layer_1_values,
                           layer_2_values,
                           desired_result,
                           self.steps_foreseen - j,
                           )


        return self

