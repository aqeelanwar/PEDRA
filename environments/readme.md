# PEDRA - Environments

PEDRA comes equip with a library of 3D realistic environments that can be used for drone applications. The environments fall into two categories.

* Indoor Environments:
  * Indoor Long Environment
  * Indoor Twist Environment
  * Indoor VanLeer Environment
  * Indoor Techno Environment
  * Indoor Pyramid Environment
  * Indoor FrogEyes Environment
  * Indoor GT Environment
  * Indoor Complex Environment
  * Indoor UpDown Environment
  * Indoor Cloud Environment


* Outdoor Environments:
  * Outdoor Courtyard
  * Outdoor Forest
  * Outdoor OldTown



## Indoor Environments:
  Screen shot and videos
  The video below is a walk through different environments so it helps you to pick based on your needs instead of downloading it and then looking at it.


## Environment file structure:
Each of the provided environments has the following three categories of files

```
|-- <environment_name>
|    |-- .exe file
|    |-- other folders
|    |-- <env_name>_floor.png
|    |-- config.cfg
```

### Unreal simulation files
These files and folder are the simulation files for packaged unreal project and are not to be modified in any way (.exe, DRLwithTL, Engine etc)

### Floorplan
A .png image file of the floorplan of the environment. This can come in handy when plotting the trajectory of the drone in the inference mode, to keep track of multiple drones in the environment and to extract drone positions in the environment.  


### Config file
Each environment comes with a config file. This config file includes the parameters used to set the environment up. Following is the list of parameters in this config file and its explanation. This config file __should not be edited__.


| Parameter        	| Explanation                                          |
|------------------	|------------------------------------------------------|
| run_name         	| Name for the current simulation                      |


Apart from the AirSim provided features, these environments have the following features



## Supported Features:

| Key        	      | Feature                                                           |Category      |
|------------------	|-------------------------------------------------------------------|--------------|
| F2         	      | Toggle PEDRA help                                                 |PEDRA         |
| P          	      | Display current position (x,y) and orientation (yaw) of the drone |PEDRA         |
| Z         	      | Toggle the floorplan minimap                                      |PEDRA         |
| 1         	      | Toggle depth map as subwindow                                     |AirSim        |
| 2         	      | Toggle segmentation map as subwindow                              |AirSim        |
| 3         	      | Toggle image from front facing camera as subwindow                |AirSim        |

More PEDRA environmental features will be added in the future releases.

<p align="center">
<img src="/images/pedra_help.gif">
</p>




## Moving around the environment (without algorithm)
Computer Vision mode and using the keys to move around. If you like any position to be used in the code hit the key P.

Add images with multiple agents



## Understanding the coordinates and conversion between them:
Three kind of coordinates
1. Physical coordinates:      The coordinates of the drone in the environment (Hitting Key P displays this coordinates)
2. Unreal Engine coordinates: The coordinates of the drone relative to the origin. This is the coordinate t
3. Image based coordinates: The coordinates of the drone in the floorplan image


Add details and images

Details on



## Extracting position of the drone in the environment
Even before running your algorithm, you might want to define some key positions for the drone for example which positions should the drone reset to after crash, what should be the goal position of the drone etc. This includes finding a suitable drone position in the environment and extracting the coordinates of this position

PEDRA provides two ways of doing that
### 1. Running PEDRA in move_around mode:
This can be done by setting the config.cfg file to move_around mode. In this mode, keyboard can be used to navigate across the environment. This mode can help the user get an idea of the environment dynamics. The keyboard keys __a, w, s, d, left, right, up and down__ can be used to navigate around. Once the user navigates the drone to position of his/her linking, key P on the keyboard can be used to display the position of the drone on the left top part of the simulation screen. Each position array displayed has three parts
```
[pitch, yaw, roll]                                # Degrees
[x coordinates, y coordinates, z coordinates]     # Physical coordinates
```


<p align="center">
<img src="/images/print_position.png">
</p>



These values can directly be fed into the orig_ip variable of the environments/initial_positions.py file in the following format making it accessible to PEDRA code
```
[x coord, y coord, yaw]
```
<p align="center">
<img src="/images/initial_positions_py.png">
</p>



### 2. Running retreive_initial_position.py:
The second way of extracting desirable drone position is to use the python file retreive_initial_position.py
```
cd PEDRA
python retrieve_initial_position.py
```
Running this will open the directory for the user to select the floorplan of the required environment. Once the user selects the floorplan, moue cursor can be used to click in the floorplan to extract the drone coordinates. All three coordinates corresponding to the position selected are displayed on the left top part of the display window.
In order to use these coordinates in the initial_positions.py file, make sure you use the physical coordinates.

<p align="center">
<img src="/images/retrieve_initial_positions.gif">
</p>


Talk about the format and all.
