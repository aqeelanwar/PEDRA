# Programmable Engine for Drone Reinforcement Learning (RL) Applications (PEDRA)

# What is PEDRA?
PEDRA is a programmable engine for Drone Reinforcement Learning (RL) applications. The engine is developed in Python and is module-wise programmable. PEDRA is targeted mainly at goal-oriented RL problems for drones, but can also be extended to other problems such as SLAM etc. The engine interfaces with Unreal gaming engine using AirSim to create the complete platform. Figure below shows the complete block diagram of the engine. [Unreal engine](https://www.google.com/search?client=safari&rls=en&q=unreal+engine&ie=UTF-8&oe=UTF-8) is used to create 3D realistic environments for the drones to be trained in. Different level of details can be added to make the environment look as realistic or as required as possible. Once the environments are ready, it is interfaced with PEDRA using [AirSim](https://github.com/microsoft/AirSim). AirSim is an open source plugin developed by Microsoft that interfaces Unreal Engine with Python. It provides basic python functionalities controlling the sensory inputs and control signals of the drone. PEDRA is built onto the low level python modules provided by AirSim creating higher level python modules for the purpose of drone RL applications.


![Cover Photo](/images/pedra_block.png)

![Cover Photo](/images/envs.png)



# PEDRA Workflow
The complete workflow of PEDRA can be seen in Figure below. The engine takes input from a config file (.cfg). This config file is used to define the problem and the algorithm for solving it. It is algorithmic specific and is used to define algorithm related parameters. Right now the supported problem is camera based autonomous navigation and the supported algorithms are single drone vanilla RL, single drone PER/DDQN based RL. More problems and associated algorithms are being added.
The most important feature of PEDRA is the high level python modules that can be used as building blocks to implement multiple algorithms for drone oriented applications. The user can either select from the above mentioned algorithms, or can create their own using these building blocks. In case the user wants to define their own problem and associated algorithm, these building blocks can be used. Once these requirements are set, the simulation can begin. PyGame screen can be used to control simulation parameters such as pausing the simulation, modifying algorithmic or training parameters, overwrite config file and save the current state of the simulation etc.  PEDRA generates a number of output files. The log file keeps track of the simulation state per iteration listing useful algorithmic parameters. This is particularly useful when troubleshooting the simulation. Tensorboard can be used to visualize the training plots in run-time. These plots are particularly useful to monitor training parameters and to change the input parameters using the PyGame screen if need be.
![Cover Photo](/images/pedra_workflow.png)


![Cover Photo](/images/depth.gif)

# Installing PEDRA
The current version of PEDRA supports Windows and requires python3. It’s advisable to [make a new virtual environment](https://towardsdatascience.com/setting-up-python-platform-for-machine-learning-projects-cfd85682c54b) for this project and install the dependencies. Following steps can be taken to download get started with PEDRA

## Clone the repository
```
git clone https://github.com/aqeelanwar/PEDRA.git
```

## Download imagenet weights for AlexNet
The DQN uses Imagenet learned weights for AlexNet to initialize the layers. Following link can be used to download the imagenet.npy file.

[Download imagenet.npy](https://drive.google.com/open?id=1Ei4mCzjfLY5ql6ILIUHaCtAR2XF6BtAM)

Once downloaded, place it in
```
models/imagenet.npy
```

## Install required packages
The provided requirements.txt file can be used to install all the required packages. Use the following command
```
cd PEDRA
pip install –r requirements.txt
```
This will install the required packages in the activated python environment.


## Install Epic Unreal Engine
You can follow the guidelines in the link below to install Unreal Engine on your platform

[Instructions on installing Unreal engine](https://docs.unrealengine.com/en-US/GettingStarted/Installation/index.html)

## Install AirSim
AirSim is an open-source plugin for Unreal Engine developed by Microsoft for agents (drones and cars) with physically and visually realistic simulations. In order to interface between Python3 and the simulated environment, AirSim needs to be installed. It can be downloaded from the link below

[Instructions on installing AirSim](https://github.com/microsoft/airsim)




# Running PEDRA
Once you have the required packages and software downloaded and running, you can take the following steps to run the code

## Create/Download a simulated environment
You can either manually create your environment using Unreal Engine, or can download one of the sample environments from the link below and run it.

[Download Environments](https://drive.google.com/open?id=1u5teth6l4JW2IXAkZAg1CbDGR6zE-v6Z)

Following environments are available for download from the link above

* Indoor Long Environment
* Indoor Twist Environment
* Indoor VanLeer Environment
* Indoor Techno Environment
* Indoor Pyramid Environment
* Indoor FrogEyes Environment
* Indoor GT Environment
* Indoor Complex Environment


The link above will help you download the packaged version of the environment for 64-bit windows. Save the folder in the unreal_env folder (create the unreal_env folder if it doesn't exist).

```
PEDRA/unreal_env/<downloaded-environment-folder>    # Generic
PEDRA/unreal_env/indoor_cloud                       # Example
```


## Edit the configuration file (Optional)
The RL parameters for the DRL simulation can be set using the provided config file and are self-explanatory. The details on the parameters in the config file can be found [here](https://towardsdatascience.com/deep-reinforcement-learning-for-drones-in-3d-realistic-environments-36821b6ee077)

```
cd PEDRA\configs
notepad config.cfg (#for windows)
```

### General Parameters [general_params]:

| Parameter        	| Explanation                                                                       	| Possible values                  	|
|------------------	|-----------------------------------------------------------------------------------	|----------------------------------	|
| run_name         	| Name for the current                                                              	| Any value                        	|
| custom_load      	| Dictates if user wants to load the network with custom weights                    	| True/False                       	|
| custom_load_path 	| If custom_load is set to True, this dictates the path of the weights to be loaded 	| Relative path to weights         	|
| env_type         	| Type of the environment (to be used in future versions)                           	| indoor/outdoor                   	|
| env_name         	| Name of the environment to be used in the simulation                              	| indoor_cloud, indoor_techno etc. 	|
| phase            	| Dictates what mode do you want to run the simulation in                           	| train / infer                    	|

### Simulation Parameters [simulation_params]:

| Parameter      	| Explanation                                           	| Possible values           	|
|----------------	|-------------------------------------------------------	|---------------------------	|
| ip_address     	| IP Address for the simulation                         	| 127.0.0.1 etc.            	|
| load_data      	| Dictates if to load data into the replay memory       	| True / False              	|
| load_data_path 	| The path to load the data from into the replay memory 	| Relative path to the data 	|

### Reinforcement Learning training parameters [RL_params]:

| Parameter              	| Explanation                                                                                     	| Possible values          	|
|------------------------	|-------------------------------------------------------------------------------------------------	|--------------------------	|
| input_size             	| The dimensions of the input image into the network                                              	| Any positive integer     	|
| num_actions            	| The size of the action space                                                                    	| 25, 400 etc              	|
| train_type             	| Dictates number of trainable layers                                                             	| e2e, last4, last3, last2 	|
| wait_before_train      	| The number of iterations to wait before traiining can begin                                     	| Any positive integer     	|
| max_iters              	| Maximum number of training iterations                                                           	| Any positive integer     	|
| buffer_len             	| The length of the replay buffer                                                                 	| Any positive integer     	|
| batch_size             	| The batch size for training                                                                     	| 8, 16, 32, 64 etc        	|
| epsilon_saturation     	| The number of iteration at which the epsilon reaches its maximum value                          	| Any positive integer     	|
| crash_thresh           	| The average depth below which the drone is considered crashed                                   	| 0.8, 1.3 etc             	|
| Q_clip                 	| Dictates if to clip the updated Q value in the Bellman equation                                 	| True, False              	|
| train_interval         	| The training happens after every train_interval iterations                                      	| 1,3,5 etc                	|
| update_target_interval 	| Copies network weights from behavior to target network every update_target_interval iterations 	| Any positive integer     	|
| gamma                  	| The value of gamma in the Bellman equation                                                      	| Between 0 and 1          	|
| dropout_rate           	| The drop out rate for the layers in the network                                                 	| Between 0 to 1           	|
| learning_rate          	| The learning rate during training                                                               	| Depends on the problem   	|
| switch_env_steps       	| The number if iterations after which to switch the initial position of the drone                	| Any positive integer     	|
| epsilon_model          	| The model used to calculate the value of epsilon for the epsilon greedy method                  	| linear, exponential      	|


## Run the Python code

### Training Phase:
To carry out training, make sure the phase parameter within the [general_params] group of the config file is set to train. After setting the parameters in under the [RL_params] category, the DRL training code can be started using the following command

```
cd PEDRA
python main.py
```

Running main.py carries out the following steps
* Attempt to load the config file
* Attempt to connect with the Unreal Engine (the indoor_long environment must be running for python to connect with the environment, otherwise connection refused warning will appear — The code won’t proceed unless a connection is established)
* Attempt to create two instances of the DNN (Double DQN is being used) and initialize them with the selected weights.
* Attempt to initialize Pygame screen for user interface
Start the DRL algorithm

At this point, the drone can be seen moving around in the environment collecting data-points. The block diagram below shows the DRL algorithm used.

![Cover Photo](/images/block_diag.png)

#### Viewing learning parameters using tensorboard
During simulation, RL parameters such as epsilon, learning rate, average Q values, loss and return can be viewed on the tensorboard. The path of the tensorboard log files depends on the env_type, env_name and train_type set in the config file and is given by
```
models/trained/<env_type>/<env_name>/Imagenet/   # Generic path
models/trained/Indoor/indoor_long/Imagenet/      # Example path
```

Once identified where the log files are stored, following command can be used on the terminal to activate tensorboard.
```
cd models/trained/Indoor/indoor_long/Imagenet/
tensorboard --logdir <train_type>                # Generic
tensorboard --logdir e2e                         # Example
```

The terminal will display the local URL that can be opened up on any browser, and the tensorboard display will appear plotting the DRL parameters on run-time.
![tensorboard](/images/tf.png)

<!-- While the simulation is running, RL parameters such as epsilon, learning rate, average Q values and loss can be viewed on the tensorboard. The path depends on the env_type, env_name and train_type set in the config file and is given by 'models/trained/&lt;env_type>/&lt;env_name>/Imagenet/''. An example can be seen below

```
cd models\trained\Indoor\indoor_long\Imagenet\
tensorboard --logdir e2e

``` -->


#### Run-time controls using PyGame screen
DRL is notorious to be data hungry. For complex tasks such as drone autonomous navigation in a realistically looking environment using the front camera only, the simulation can take hours of training (typically from 8 to 12 hours on a GTX1080 GPU) before the DRL can converge. In the middle of the simulation, if you feel that you need to change a few DRL parameters, you can do that by using the PyGame screen that appears during your simulation. This can be done using the following steps
1. Change the config file to reflect the modifications (for example decrease the learning rate) and save it.
2. Select the Pygame screen, and hit ‘backspace’. This will pause the simulation.
3. Hit the ‘L’ key. This will load the updated parameters and will print it on the terminal.
4. Hit the ‘backspace’ key to resume the simulation.
Right now the simulation only updates the learning rate. Other variables can be updated too by editing the aux_function.py file for the module check_user_input



### Inference Phase:
To run the simulation in the inference phase, make sure the phase parameter within the [general_params] group of the config file is set to infer. Custom weights can be loaded into the network by setting the following parameters

```
custom_load_path: True
custom_load_path: <path_to_weights>
```

#### Output graphs
The simulation updates two graphs in real-time. The first graph is the altitude variation of the drone, while the other one is the drone trajectory mapped onto the environment floorplan. The trajectory graph also reports the total distance traveled by the drone before crash.

![Inference graphs](/images/infer.gif)

#### Run-time controls using PyGame screen
Right now the simulation supports only the following two functionalities (other functionalities can be added by modifying the check_user_input module in the aux_function.py file for the phase infer)

* Backspace key: Pause/Unpause the simulation
* S key: Save the altitude variation and trajectory graphs at the following location

```
unreal_env/<env_name>/results/
```

# Example: Deep Reinforcement Learning with Transfer Learning (DRLwithTL-Sim)

DRLwithTL is a transfer learning based approach to reduce on-board computation required to train a deep neural network for autonomous navigation via Deep Reinforcement Learning for a target algorithmic performance. PEDRA provided environments are used to train the network on a set of meta-environments. These trained meta-weights are then used as initializers to the network in test environments and fine-tuned for the last few fully connected layers. Variation in drone dynamics and environmental characteristics is carried out to show robustness of the approach. The repository containing the code for real environment on a real DJI Tello drone can be found @ [DRLwithTL-Real](https://github.com/aqeelanwar/DRLwithTL_real)
## Introductory Video
[![Watch the video](/images/video_cover.png)](https://youtu.be/zmR0KB_qle8)

## PEDRA config for DRLwithTL
PEDRA's config file can be used to carry out DRLwithTL. The parameter train_type can be used to dictate how many layers from the end needs to be trained.


# Citing
If you find this repository useful for your research please use the following bibtex citations

```
@ARTICLE{2019arXiv191005547A,
       author = {Anwar, Aqeel and Raychowdhury, Arijit},
        title = "{Autonomous Navigation via Deep Reinforcement Learning for Resource Constraint Edge Nodes using Transfer Learning}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Statistics - Machine Learning},
         year = "2019",
        month = "Oct",
          eid = {arXiv:1910.05547},
        pages = {arXiv:1910.05547},
archivePrefix = {arXiv},
       eprint = {1910.05547},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv191005547A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
```
@article{yoon2019hierarchical,
  title={Hierarchical Memory System With STT-MRAM and SRAM to Support Transfer and Real-Time Reinforcement Learning in Autonomous Drones},
  author={Yoon, Insik and Anwar, Malik Aqeel and Joshi, Rajiv V and Rakshit, Titash and Raychowdhury, Arijit},
  journal={IEEE Journal on Emerging and Selected Topics in Circuits and Systems},
  volume={9},
  number={3},
  pages={485--497},
  year={2019},
  publisher={IEEE}
}
```

## Authors
* [Aqeel Anwar](https://www.prism.gatech.edu/~manwar8) - Georgia Institute of Technology

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
