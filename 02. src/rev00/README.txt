README

This is the working version rev00. Result is a path generated, but the execution time is very long.

Data Structures:
- Priority queue for managing the open list during A* search.
- Set to store visited nodes information.
- Numpy array to represent the map and obstacles.

Comparing Duplicate Nodes:
- A set data structure is utilized to compare duplicate nodes efficiently based on their indices.
- Set size in xy direction is half of the config space. 

Heuristic Function Method:
- Euclidean distance is employed as the heuristic function to estimate the distance between nodes.

Cost to Come:
- The cost to come for each node is calculated based on the step size and movement actions.
- Cost to come is set up to increase turning cost more than going straight

Map Creation Method:
- Obstacles are represented using mathematical equations.
- The map is created by setting obstacle positions and adding borders.
- Obstacles are represented as coordinates, returned in obstacle_positions variable.

Obstacle Check:
- A function checks whether a node is obstructed by an obstacle efficiently. Checked during the A* search.
