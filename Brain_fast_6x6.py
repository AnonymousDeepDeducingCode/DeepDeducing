import numpy as np
from scipy.special import expit


class Brain(object):
    def __init__(self, dims, alpha, epochs, tilt_1, variation_1, update_rate_1, beta, rounds, update_rate_2):

        self.dims                         = dims
        self.number_of_layers             = self.dims.shape[0]

        self.alpha                        = alpha
        self.epochs                       = epochs

        self.tilt_1                       = tilt_1
        self.variation_1                  = variation_1
        self.update_rate_1                = update_rate_1


        self.beta                         = beta
        self.rounds                       = rounds

        self.update_rate_2                = update_rate_2

        self.synapse_list                 = self.initialize_weights()

        self.tilt_1_list                  = self.initialize_tilt_1()


    def initialize_weights(self):
        synapse_list = list()
        for i in range(self.number_of_layers - 1):
            synapse                                              = (np.random.random((self.dims[i]                                 , self.dims[i+1]                          )) -0.5 ) * 0.1
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

            self.synapse_list[i]                            += np.atleast_2d(layer_list[i]                                                          ).T.dot(np.array(layer_delta_list[- 1 - i])                            ) * self.alpha
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


    def generate_batch_values_for_each_layer(self, input):

        layer_list                = list()

        layer                     = input

        layer_list.append(layer)

        for i in range(self.number_of_layers - 1):

            layer                 = self.activator((layer_list[-1]                         .dot( self.synapse_list[i]                                                          ) ) * self.tilt_1_list[i])

            layer_list.append(layer)

        return   layer_list


    def train_for_input(self,
                       layer_list):


        layer_final_error      = np.ones_like(layer_list[-1]) - layer_list[-1]

        layer_delta            = layer_final_error                                                                                              * self.activator_output_to_derivative(layer_list[-1])           * self.tilt_1_list[-1]


        for i in range(self.number_of_layers - 2):

            layer_delta        = (layer_delta.dot( self.synapse_list[- 1 - i].T                                                         ) )     * self.activator_output_to_derivative(layer_list[- 1 - 1 - i])  * self.tilt_1_list[-1 -1 -i]

        tilt_2_delta       = (layer_delta.dot( self.synapse_list[0].T                                                                   ) )     * self.activator_output_to_derivative(layer_list[0])            * self.sudoku_matrix_inner_batch
        layer_delta        = (layer_delta.dot( self.synapse_list[0].T                                                                   ) )     * self.activator_output_to_derivative(layer_list[0])            * self.tilt_2_list_batch

        self.sudoku_matrix_inner_batch_update = layer_delta  * self.beta          * self.sudoku_matrix_resistor_batch
        self.tilt_2_list_batch_update         = tilt_2_delta * self.update_rate_2 * self.sudoku_matrix_resistor_batch


    def deduce_from(self, sudoku_matrix_inner, tilt_2_list, sudoku_matrix_resistor, goal):

        self.sudoku_matrix_inner    = sudoku_matrix_inner
        self.sudoku_size            = self.sudoku_matrix_inner.shape[0]
        self.tilt_2_list            = tilt_2_list
        self.sudoku_matrix_resistor = sudoku_matrix_resistor

        for i in range(self.rounds):

            self.sudoku_matrix_inner_batch    =   np.array([self.sudoku_matrix_inner[0, :].flatten(), self.sudoku_matrix_inner[1, :].flatten(), self.sudoku_matrix_inner[2, :].flatten(), self.sudoku_matrix_inner[3, :].flatten(), self.sudoku_matrix_inner[4, :].flatten(), self.sudoku_matrix_inner[5, :].flatten(),
                                                            self.sudoku_matrix_inner[:, 0].flatten(), self.sudoku_matrix_inner[:, 1].flatten(), self.sudoku_matrix_inner[:, 2].flatten(), self.sudoku_matrix_inner[:, 3].flatten(), self.sudoku_matrix_inner[:, 4].flatten(), self.sudoku_matrix_inner[:, 5].flatten()])

            self.tilt_2_list_batch            =   np.array([self.tilt_2_list[0, :].flatten(), self.tilt_2_list[1, :].flatten(), self.tilt_2_list[2, :].flatten(), self.tilt_2_list[3, :].flatten(), self.tilt_2_list[4, :].flatten(), self.tilt_2_list[5, :].flatten(),
                                                            self.tilt_2_list[:, 0].flatten(), self.tilt_2_list[:, 1].flatten(), self.tilt_2_list[:, 2].flatten(), self.tilt_2_list[:, 3].flatten(), self.tilt_2_list[:, 4].flatten(), self.tilt_2_list[:, 5].flatten()])

            self.sudoku_matrix_resistor_batch =   np.array([self.sudoku_matrix_resistor[0, :].flatten(), self.sudoku_matrix_resistor[1, :].flatten(), self.sudoku_matrix_resistor[2, :].flatten(), self.sudoku_matrix_resistor[3, :].flatten(), self.sudoku_matrix_resistor[4, :].flatten(), self.sudoku_matrix_resistor[5, :].flatten(),
                                                            self.sudoku_matrix_resistor[:, 0].flatten(), self.sudoku_matrix_resistor[:, 1].flatten(), self.sudoku_matrix_resistor[:, 2].flatten(), self.sudoku_matrix_resistor[:, 3].flatten(), self.sudoku_matrix_resistor[:, 4].flatten(), self.sudoku_matrix_resistor[:, 5].flatten()])


            layer_list  = self.generate_batch_values_for_each_layer(self.activator( self.sudoku_matrix_inner_batch * self.tilt_2_list_batch  ))


            self.train_for_input(layer_list)

            self.sudoku_matrix_inner[0, :] += self.sudoku_matrix_inner_batch_update[0 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[1, :] += self.sudoku_matrix_inner_batch_update[1 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[2, :] += self.sudoku_matrix_inner_batch_update[2 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[3, :] += self.sudoku_matrix_inner_batch_update[3 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[4, :] += self.sudoku_matrix_inner_batch_update[4 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[5, :] += self.sudoku_matrix_inner_batch_update[5 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[:, 0] += self.sudoku_matrix_inner_batch_update[6 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[:, 1] += self.sudoku_matrix_inner_batch_update[7 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[:, 2] += self.sudoku_matrix_inner_batch_update[8 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[:, 3] += self.sudoku_matrix_inner_batch_update[9 ].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[:, 4] += self.sudoku_matrix_inner_batch_update[10].reshape((self.sudoku_size, self.sudoku_size))
            self.sudoku_matrix_inner[:, 5] += self.sudoku_matrix_inner_batch_update[11].reshape((self.sudoku_size, self.sudoku_size))

            self.tilt_2_list[0, :]         += self.tilt_2_list_batch_update[0 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[1, :]         += self.tilt_2_list_batch_update[1 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[2, :]         += self.tilt_2_list_batch_update[2 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[3, :]         += self.tilt_2_list_batch_update[3 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[4, :]         += self.tilt_2_list_batch_update[4 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[5, :]         += self.tilt_2_list_batch_update[5 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[:, 0]         += self.tilt_2_list_batch_update[6 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[:, 1]         += self.tilt_2_list_batch_update[7 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[:, 2]         += self.tilt_2_list_batch_update[8 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[:, 3]         += self.tilt_2_list_batch_update[9 ].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[:, 4]         += self.tilt_2_list_batch_update[10].reshape((self.sudoku_size, self.sudoku_size))
            self.tilt_2_list[:, 5]         += self.tilt_2_list_batch_update[11].reshape((self.sudoku_size, self.sudoku_size))


        return self.sudoku_matrix_inner, self.tilt_2_list


