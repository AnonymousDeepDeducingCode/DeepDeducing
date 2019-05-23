# Suggestion

      Software Suggestion:
            Interpreter: Python 3.7
            Library:     numpy 1.16.2
                         scipy 1.2.1
      Hardware Suggestion:
            CPU:         Intel i9-9900k 
            OC:          4.5~5.0 ghz
      
# Content

.py

      Generate_Synapse.py---- Train the neural network and generate/save matrix weights (learning phase)
      Run.py             ---- Load saved matrix weight and sovle Sudoku tables (deducing phase)
      Brain.py           ---- Module to be imported in Generate_Synapse.py/Run.py. Might be slower than Brain_fast_6x6.py
      Brain_fast_6x6.py  ---- Module to be imported in Generate_Synapse.py/Run.py. for faster reproduction

.npy

      self.synapse_list_6x6_100x100x100_0.000001_400m_tilt_30--- 100x100x100 means the topology of the neural network, 0.001 means learning rate, 400m means its been trained for 400 million iterations, tilt_30 means the slope of sigmoid function is set to be 30 and will be updated and fine-tuned
      self.tilt_1_list_6x6_100x100x100_0.000001_400m_tilt_30 --- tilt_1_list means the tilt list for the sigmoid function in each layer (except input layer) of the neural network
      
# Others
The codes are anonymous as well.
