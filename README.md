## A* Path Planning Algorithm

This project implements the A* algorithm for path planning in a 2D grid environment with obstacles. The goal is to find the optimal path from a given start position to a goal position while avoiding obstacles.

## Preview output video
[Google Drive link](https://drive.google.com/file/d/1jdlul5UnlfNsAlWaxlKJbDAy2I5n2Y5F/view?usp=drive_link)

### Map Configuration
The map is represented as a 2D grid with dimensions of 1200x500 units. The grid is divided into cells, and each cell can be either free space or an obstacle.
The map boundaries are also considered as obstacles.

### Dependencies
The project requires the following dependencies:
- Python 3.x
- NumPy
- OpenCV (cv2)

You can install the dependencies using the following command:
```bash
pip install numpy opencv-python
```

### Running the Code
The latest version is the revision with the highest number
To run the path planning algorithm, execute the following command:
```bash
python a_star_hoang_fazil.py
```
The program will prompt you to enter the following information:
- Start position (x, y, orientation)
- Goal position (x, y, orientation)
- Step size increment (1-10)

The start and goal positions should be entered in the format x y orientation, where x and y are the coordinates, and orientation is the angle in degrees (0-360) in multiples of 30.

### Methods Used to Accelerate Path Finding
Several methods and optimizations have been implemented to accelerate the path_finding process:

- **Cost Grid:** A grid is used to store the cost of reaching each node, allowing for efficient updates of costs when a better path to a node is found.
- **Priority Queue:** A priority queue prioritizes nodes based on their total cost, guiding the search towards the most promising paths.
- **Heuristic Function:** The diagonal distance heuristic estimates the cost from a node to the goal node, providing a more informed estimate compared to the Euclidean distance.
- **Pruning Duplicate Nodes:** Duplicate nodes are checked using the same cost_grid with total cost values, reducing redundant explorations.
- **Visualization Optimization:** Visualization of explored nodes and their action vectors is optimized by drawing only every 500th node, improving performance without significantly impacting the visual representation.

### Results
The program visualizes the explored nodes, their corresponding action vectors, and the optimal path (if found) using OpenCV.

The explored nodes are represented by pick arrows, indicating the possible actions from each node. The optimal path is represented by yellow arrows, connecting the start position to the goal position.

Additionally, the program prints the number of steps in the optimal path and the time taken for path discovery.
