import numpy as np
import random
import copy

present_maps_and_player_movements       = list()
consequence_maps                        = list()

#============== PHASE ==================================================================================================

present_map     = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())

#===========================================================================================================

present_map     = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())


#===========================================================================================================

present_map     = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())


#===========================================================================================================

present_map     = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())


#======================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())




#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())



#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())

#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())


#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())


#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())



#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())



#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())


#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())




#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())




#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())



#===========================================================================================================

present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
player_movement = np.array( [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
player_movement = np.array( [0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
player_movement = np.array( [0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())
present_map     = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
player_movement = np.array( [0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
consequence_map = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 1]])
present_maps_and_player_movements.append(np.concatenate( (present_map.flatten(), player_movement) ))
consequence_maps.append                 (consequence_map.flatten())









#============== SUMMING UP ========================================================================================

present_maps_and_player_movements = copy.deepcopy(present_maps_and_player_movements)
consequence_maps                  = copy.deepcopy(consequence_maps)

present_maps_and_player_movements = np.array(present_maps_and_player_movements)
consequence_maps                  = np.array(consequence_maps)

#============== IMPORTING MODEL ======================================================================================

from Game_Net_Crow_Reasoning_particle_swarm_x6 import *

input_dim      = consequence_map.shape[0] * consequence_map.shape[1] + player_movement.shape[0]                         # ============= BE WARE OF DIMENSION =============================
output_dim     = consequence_map.shape[0] * consequence_map.shape[1]
alpha          = 0.01                                                                                                    # 0.1
epochs         =  100000 * consequence_maps.shape[0]                                                                      # 3000
beta           = 0.5
rounds         = 25000
Machine        = Game_Net_Problem_Solving(input_dim, output_dim, alpha, epochs, beta, rounds)

#============== TRAINING MODEL =======================================================================================




for i in range(Machine.epochs):
    random.seed()
    random_index                    = np.random.randint(present_maps_and_player_movements.shape[0])
    present_map_and_player_movement = present_maps_and_player_movements[random_index]
    consequence_map                 = consequence_maps[random_index]
    Machine.fit(present_map_and_player_movement, consequence_map)


Machine.DFNN_to_RNN(10, 25)
np.savetxt("self.synapse_layer_3_to_future_layer_1.csv"     , Machine.synapse_layer_3_to_future_layer_1  , delimiter=",")
np.savetxt("self.conduit_layer_3_to_future_layer_3.csv"     , Machine.conduit_layer_3_to_future_layer_3  , delimiter=",")
np.savetxt("self.synapse_layer_0_to_layer_1.csv"           , Machine.synapse_layer_0_to_layer_1         , delimiter=",")
np.savetxt("self.synapse_layer_1_to_layer_2.csv"           , Machine.synapse_layer_1_to_layer_2         , delimiter=",")
np.savetxt("self.synapse_layer_2_to_layer_3.csv"           , Machine.synapse_layer_2_to_layer_3         , delimiter=",")

