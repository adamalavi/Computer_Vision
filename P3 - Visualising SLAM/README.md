# Project : Landmark Detection & Robot Tracking (SLAM)

## Description

In this project, we implement SLAM (Simultaneous Localization and Mapping for a 2 dimensional world.
We combine robot sensor measurements and movement to create a map of an environment from only sensor and motion
data gathered by a robot, over time. 

SLAM gives us a way to track the location of a robot in the world in real-time and identify the locations of landmarks such as buildings, trees, rocks, and other world features.

This is an active area of research in the field of robotics and autonomous systems.

Below is an example of a 2D robot world with landmarks (purple x's) and the robot (a red 'o') located and found using only sensor and motion data collected by that robot.

<p align="center">
  <img src="https://github.com/adamalavi/Computer_Vision/blob/master/P3%20-%20Visualising%20SLAM/images/description.png">
</p>

## Graph SLAM

To implement Graph SLAM, a matrix and a vector (omega and xi, respectively) are introduced. 
The matrix is square and labelled with all the robot poses (xi) and all the landmarks (Li).

Every time we make an observation, for example as we move between two poses by some distance
`dx` and can relate those two positions, you can represent this as a numerical relationship in these
matrices.

We are referring to robot poses as Px, Py and landmark positions as Lx, Ly, and one way to approach this challenge is to add both x and y locations in the constraint matrices.

<p align="center">
  <img src="https://github.com/adamalavi/Computer_Vision/blob/master/P3%20-%20Visualising%20SLAM/images/constraints2D.png" width=50% height=50%>
</p>

### Implementation

After correctly implementing SLAM, we can visualize 

- Omega : 
<p align="center">
  <img src="https://github.com/adamalavi/Computer_Vision/blob/master/P3%20-%20Visualising%20SLAM/images/implement_omega.png">
</p>

- Xi : 

<p align="center">
  <img src="https://github.com/adamalavi/Computer_Vision/blob/master/P3%20-%20Visualising%20SLAM/images/implement_xi.png">
</p>

## Solution 
To "solve" for all these x values, we can use linear algebra; all the values of x are in the vector `mu` which can be calculated as a product of the inverse of omega times `xi`.

<p align="center">
  <img src="https://github.com/adamalavi/Computer_Vision/blob/master/P3%20-%20Visualising%20SLAM/images/solution_clean.png">
</p>

The result can be displayed as well : 

<p align="center">
  <img src="https://github.com/adamalavi/Computer_Vision/blob/master/P3%20-%20Visualising%20SLAM/images/implement_mu.png">
</p>

Notice that our initial position is `[x = world_size / 2, y = world_size / 2]` since we assume
with absolute confidence that our robot starts out in the middle of the defined world.

## Visualization of the constructed world

Finally, we can visualize the code implemented : the final position of the robot
and the position of landmarks, created from only motion and measurement data ! 

<p align="center">
  <img src="https://github.com/adamalavi/Computer_Vision/blob/master/P3%20-%20Visualising%20SLAM/images/constructed_world.png">
</p>

