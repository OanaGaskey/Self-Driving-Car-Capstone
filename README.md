# Programming a Real Self-Driving Car

## System Integration Project

[image1]: ./imgs/Carla.png "Image 1"     

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree. The goal of the project is to fully implement with ROS the main modules of an autonomous vehicle: Perception, Planning and Control.

The project was designed using a simulator where the car drives around a highway test track with traffic lights. The simulator can be found [here] (https://github.com/udacity/CarND-Capstone/releases) and starter code [here] (https://github.com/udacity/CarND-Capstone) 


## The Team

|           | Name                     |    E-Mail                        |      GitHub                                     |
| --------- | -------------------------| -------------------------------- | :----------------------------------------------:|
| Team Lead | Mark Draghicescu         |    markd38@hotmail.com           |      https://github.com/mark-draghicescu        |
|           | Oana Gaskey              |    oana.gaskey@gmail.com         |      https://github.com/OanaGaskey              |
|           | Ayomide Yusuf            |    mailayomide@gmail.com         |      https://github.com/ayomide-adekunle        |
|           | Rajat Roy                |    rctbraj@gmail.com             |      https://github.com/Rajat-Roy               |


## Overview

[image1]: architecture
The submitted code is implemented in ROS. For this project we mainly use __rospy__, which is a pure Python client library for ROS and enables Python programmers to interface with ROS Topics, Services and Parameters.

## Code Architecture
1. Perception Module
     * 1.1 Traffic Light Detection Node
     
2. Planning Module
      * 2.1 Waypoint Loader
      * 2.2 Waypoint Updater
      
3. Control Module
      * 3.1 DBW Node
      * 3.2 Waypoint Follower

### 1. Perception Module

### 2. Planning Module

### 3. Control Module

## Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
