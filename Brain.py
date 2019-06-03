import numpy as np
from scipy.special import expit

class Brain(object):
    def __init__(self, dims, tilt_1, variation_1, update_rate_1, variation_W, update_rate_W, epochs, update_rate_2, update_rate_inner, rounds):

        self.dims                         = dims
        self.number_of_layers             = self.dims.shape[0]

        self.tilt_1                       = tilt_1
        self.variation_1                  = variation_1
        self.update_rate_1                = update_rate_1

        self.variation_W                  = variation_W
        self.update_rate_W                = update_rate_W

        self.epochs                       = epochs

        self.update_rate_2                = update_rate_2

        self.update_rate_inner            = update_rate_inner

        self.rounds                       = rounds

        self.synapse_list                 = self.initialize_weights()

        self.tilt_1_list                  = self.initialize_tilt_1()


    def initialize_weights(self):
        synapse_list = list()
        for i in range(self.number_of_layers - 1):
            synapse                                              = (np.random.random((self.dims[i]                                 , self.dims[i+1]                          )) -0.5 ) * self.variation_W
            synapse_list.append(synapse)
        synapse_list = np.asarray(synapse_list)
        return  synapse_list


    def initialize_tilt_1(self):
        tilt_list = list()
        for i in range(self.number_of_layers - 1):
            tilt                                                 = (np.random.random((self.dims[i + 1]                             ,                                         )) -0.5 ) * self.variation_1 + self.tilt_1
            tilt_list.append(tilt)
        tilt_list = np.asarray(tilt_list)
        return tilt_list


    def activator(self, x):
        return expit(x)


    def activator_output_to_derivative(self, output):
        return output * (1 - output)


    def generate_values_for_each_layer(self, input):

        layer_list                = list()

        layer                     = input

        layer_list.append(layer)

        for i in range(self.number_of_layers - 2):

            layer                 = self.activator(np.dot(layer_list[-1]                          , self.synapse_list[i]                                                          ) * self.tilt_1_list[i] )

            layer_list.append(layer)

        layer                     = self.activator(np.dot(layer_list[-1]                          , self.synapse_list[-1]                                                         ) * self.tilt_1_list[-1] )

        layer_list.append(layer)

        return   np.array(layer_list)


    def train_for_each(self,
                       layer_list,
                       output):

        layer_delta_list       = list()
        tilt_1_delta_list      = list()

        layer_final_error      = np.array([output - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                               )    * np.array([self.activator_output_to_derivative(layer_list[-1] )           * self.tilt_1_list[-1]                                              ])
        tilt_1_delta           = (layer_final_error                                                                                               )[0] * np.array( self.activator_output_to_derivative(layer_list[-1] )           * np.dot(layer_list[-2], self.synapse_list[-1])                      )

        layer_delta_list.append(layer_delta)
        tilt_1_delta_list.append(tilt_1_delta)

        for i in range(self.number_of_layers - 2):

            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) )    * np.array([self.activator_output_to_derivative(layer_list[- 1 - 1 - i] )  * self.tilt_1_list[-1 -1 -i]                                        ])
            tilt_1_delta       = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                         ) )[0] * np.array( self.activator_output_to_derivative(layer_list[- 1 - 1 - i] )  * np.dot(layer_list[-1 -1 -i -1], self.synapse_list[-1 -1 -i])       )

            layer_delta_list.append(layer_delta)
            tilt_1_delta_list.append(tilt_1_delta)

        for i in range(self.number_of_layers - 1):

            self.synapse_list[i]                            += np.atleast_2d(layer_list[i]                                                          ).T.dot(np.array(layer_delta_list[- 1 - i])                            ) * self.update_rate_W
            self.tilt_1_list[i]                             += np.array(tilt_1_delta_list[- 1 - i])                                                                                                                          * self.update_rate_1


    def fit(self, input_list, output_list):

        input_list  = np.array(input_list)
        output_list = np.array(output_list)

        for i in range(self.epochs):

            print(i)

            random_index = np.random.randint(input_list.shape[0])
            input        = input_list[random_index]
            output       = output_list[random_index]

            layer_list = self.generate_values_for_each_layer(input)

            self.train_for_each(
                       layer_list,
                       output)

        return self


#-----------------------------------------------------------------------Deducing Phase Functions------------------------------------------------------------------------------------------------------------


    def train_for_input(self,
                       layer_list,
                       index,
                       position,
                       goal):

        layer_delta_list       = list()

        layer_final_error      = np.array([goal - layer_list[-1]])

        layer_delta            = (layer_final_error                                                                                                ) * np.array([self.activator_output_to_derivative(layer_list[-1]                     ) * self.tilt_1_list[-1] ])

        layer_delta_list.append(layer_delta)

        for i in range(self.number_of_layers - 2):

            layer_delta           = (layer_delta_list[-1].dot(self.synapse_list[- 1 - i].T                                                       ) ) * np.array([self.activator_output_to_derivative(layer_list[- 1 - 1 - i]            ) * self.tilt_1_list[-1-1-i]                             ])

            layer_delta_list.append(layer_delta)

        if position == "row":
            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[0].T                                                            ) ) * np.array([self.activator_output_to_derivative(layer_list[0]                      ) * self.tilt_2_list[index, :].flatten()                 ])
            tilt_2_delta       = (layer_delta_list[-1].dot(self.synapse_list[0].T                                                            ) ) * np.array( self.activator_output_to_derivative(layer_list[0]                      ) * self.sodoku_matrix_inner[index, :].flatten()          )

        if position == "column":
            layer_delta        = (layer_delta_list[-1].dot(self.synapse_list[0].T                                                            ) ) * np.array([self.activator_output_to_derivative(layer_list[0]                      ) * self.tilt_2_list[:, index].flatten()                 ])
            tilt_2_delta       = (layer_delta_list[-1].dot(self.synapse_list[0].T                                                            ) ) * np.array( self.activator_output_to_derivative(layer_list[0]                      ) * self.sodoku_matrix_inner[:, index].flatten()          )

        layer_delta_list.append(layer_delta)

        if position == "row":
            self.sodoku_matrix_inner[index, :] += (np.array(layer_delta_list[-1][0]) * self.update_rate_inner * self.sodoku_matrix_resistor[index, :].flatten()   ).reshape((self.sodoku_size, self.sodoku_size))
            self.tilt_2_list[index, :]         += (np.array(tilt_2_delta)            * self.update_rate_2     * self.sodoku_matrix_resistor[index, :].flatten()   ).reshape((self.sodoku_size, self.sodoku_size))

        if position == "column":
            self.sodoku_matrix_inner[:, index] += (np.array(layer_delta_list[-1][0]) * self.update_rate_inner * self.sodoku_matrix_resistor[:, index].flatten()   ).reshape((self.sodoku_size, self.sodoku_size))
            self.tilt_2_list[:, index]         += (np.array(tilt_2_delta)            * self.update_rate_2     * self.sodoku_matrix_resistor[:, index].flatten()   ).reshape((self.sodoku_size, self.sodoku_size))


    def deduce_from(self, sodoku_matrix_inner, tilt_2_list, sodoku_matrix_resistor, goal):

        self.sodoku_matrix_inner    = sodoku_matrix_inner
        self.sodoku_size            = self.sodoku_matrix_inner.shape[0]
        self.tilt_2_list            = tilt_2_list
        self.sodoku_matrix_resistor = sodoku_matrix_resistor

        for i in range(self.rounds):

            layer_lists = list()

            for j in range(self.sodoku_size * 2):

                if j + 1 <= self.sodoku_size:
                    layer_list  = self.generate_values_for_each_layer(self.activator(self.sodoku_matrix_inner[j, :].flatten()                    * self.tilt_2_list[j, :].flatten()                     ))
                    layer_lists.append(layer_list)
                else:
                    layer_list  = self.generate_values_for_each_layer(self.activator(self.sodoku_matrix_inner[:, j - self.sodoku_size].flatten() * self.tilt_2_list[:, j - self.sodoku_size].flatten()  ))
                    layer_lists.append(layer_list)


            for k in range(self.sodoku_size * 2):

                if k + 1 <= self.sodoku_size:
                    self.train_for_input(layer_lists[k],
                                         k,
                                         "row",
                                         goal)
                else:
                    self.train_for_input(layer_lists[k],
                                         k - self.sodoku_size,
                                         "column",
                                         goal)

        return self.sodoku_matrix_inner, self.tilt_2_list


