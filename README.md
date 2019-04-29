# Suggestion

      Software Suggestion:
            Interpreter: Python 3.7
            Library:     numpy 1.16.2
      Hardware Suggestion:
            CPU:         Intel i9-9900k 
            OC:          4.5~5.0 ghz
      
# Content

.py

      Generate_Synapse.py---- Train the neural network and generate/save matrix weights (learning phase)
      Run.py             ---- Load saved matrix weight and sovle Sudoku tables (deducing phase)
      Brain.py           ---- Module to be imported in Generate_Synapse.py/Run.py. Might be slower than Brain_fast_6x6.py
      Brain_fast_6x6.py  ---- Module to be used to train/solve Sudoku tables of size 6x6 for faster reproduction

.npy

      self.synapse_list_6x6_500x500x500_0.0001_25m_tilt_30_flexible---- 500x500x500 means the topology of the neural network, 25m means its been trained for 25 million iterations, 0.001 means alpha, tilt_30_flexible means the slope of sigmoid function is set to be 30 and will be updated
      self.tilt_list_6x6_500x500x500_0.0001_25m_tilt_30_flexible   ---- tilt_list means the tilt list for the sigmoid function in each layer of the neural network
      
# Others
The codes are anonymous as well.
