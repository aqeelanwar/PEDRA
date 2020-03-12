# PEDRA - Provided Algorithms
## 1. Deep Q-learning
Value based deep Q learning with Double DQN and prioritized experienced replay

### Block Diagram
![Cover Photo](/images/block_diag.png)


### Simulation Parameters [simulation_params]:

| Parameter      	| Explanation                                           	| Possible values           	|
|----------------	|-------------------------------------------------------	|---------------------------	|
| load_data      	| Dictates if to load data into the replay memory       	| True / False              	|
| load_data_path 	| The path to load the data from into the replay memory 	| Relative path to the data 	|

### Reinforcement Learning training parameters [RL_params]:

| Parameter              	| Explanation                                                                                     	| Possible values          	|
|------------------------	|-------------------------------------------------------------------------------------------------	|--------------------------	|
| input_size             	| The dimensions of the input image into the network                                              	| Any positive integer     	|
| num_actions            	| The size of the action space                                                                    	| 25, 400 etc              	|
| train_type             	| Dictates number of trainable layers                                                             	| e2e, last4, last3, last2 	|
| wait_before_train      	| The number of iterations to wait before training can begin                                     	  | Any positive integer     	|
| max_iters              	| Maximum number of training iterations                                                           	| Any positive integer     	|
| buffer_len             	| The length of the replay buffer                                                                 	| Any positive integer     	|
| batch_size             	| The batch size for training                                                                     	| 8, 16, 32, 64 etc        	|
| epsilon_saturation     	| The number of iteration at which the epsilon reaches its maximum value                          	| Any positive integer     	|
| crash_thresh           	| The average depth below which the drone is considered crashed                                   	| 0.8, 1.3 etc             	|
| Q_clip                 	| Dictates if to clip the updated Q value in the Bellman equation                                 	| True, False              	|
| train_interval         	| The training happens after every train_interval iterations                                      	| 1,3,5 etc                	|
| update_target_interval 	| Copies network weights from behavior to target network every update_target_interval iterations 	  | Any positive integer     	|
| gamma                  	| The value of gamma in the Bellman equation                                                      	| Between 0 and 1          	|
| dropout_rate           	| The drop out rate for the layers in the network                                                 	| Between 0 to 1           	|
| learning_rate          	| The learning rate during training                                                               	| Depends on the problem   	|
| switch_env_steps       	| The number if iterations after which to switch the initial position of the drone                	| Any positive integer     	|
| epsilon_model          	| The model used to calculate the value of epsilon for the epsilon greedy method                  	| linear, exponential      	|


## Download imagenet weights for AlexNet
The DQN uses Imagenet learned weights for AlexNet to initialize the layers. Following link can be used to download the imagenet.npy file.

[Download imagenet.npy](https://drive.google.com/open?id=1Ei4mCzjfLY5ql6ILIUHaCtAR2XF6BtAM)

Once downloaded, place it in
```
models/imagenet.npy
```

#### Run-time controls using PyGame screen
DRL is notorious to be data hungry. For complex tasks such as drone autonomous navigation in a realistically looking environment using the front camera only, the simulation can take hours of training (typically from 8 to 12 hours on a GTX1080 GPU) before the DRL can converge. In the middle of the simulation, if you feel that you need to change a few DRL parameters, you can do that by using the PyGame screen that appears during your simulation. This can be done using the following steps
1. Change the config file to reflect the modifications (for example decrease the learning rate) and save it.
2. Select the Pygame screen, and hit ‘backspace’. This will pause the simulation.
3. Hit the ‘L’ key. This will load the updated parameters and will print it on the terminal.
4. Hit the ‘backspace’ key to resume the simulation.
Right now the simulation only updates the learning rate. Other variables can be updated too by editing the aux_function.py file for the module check_user_input



### Inference Mode:
To run the simulation in the inference mode, make sure the mode parameter within the [general_params] group of the config file is set to infer. Custom weights can be loaded into the network by setting the following parameters

```
custom_load_path: True
custom_load_path: <path_to_weights>
```


#### Run-time controls using PyGame screen
Right now the simulation supports only the following two functionalities (other functionalities can be added by modifying the check_user_input module in the aux_function.py file for the mode infer)

* Backspace key: Pause/Unpause the simulation
* S key: Save the altitude variation and trajectory graphs at the following location

```
unreal_env/<env_name>/results/
```
