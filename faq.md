
# FAQs


### *How to install AirSim plugin?*
AirSim is an open-source plugin for Unreal Engine developed by Microsoft for agents (drones and cars) with physically and visually realistic simulations. In case you decide on creating your own environments on Unreal Engine (and not use the ones provided for download) you need to install the plugin into Unreal Engine. Details on how to install the plugin can be found below.

[Instructions on installing AirSim](https://github.com/microsoft/airsim)

### *How to set up initial positions in an environment:*
Following module can be used to dictate initial positions for drone in the environment
```
unreal_envs/initial_positions.py
```

1. Locate the python module with the name of the environment and add to the orig_ip array. Each member of the orig_ip array is one initial position corresponding to (x, y, theta) where x and y are the positional coordinates and theta is the orientation (yaw). Make sure that you don't modify the first initial position commented as __Player start__.
2. In order to add a position from the environment, set the mode in teh config file to move_around
```
mode:     move_around
```
3. Run PEDRA
```
python main.py
```
4. Navigate to the desired position using the keyboard keys w, a, s, dup, down, left and right
5. Hit the key 'P'. The unreal coordinates of the current position will be shown at the left top of the screen

More details can be found [here](/unreal_envs/readme.md)
