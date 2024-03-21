# Description: This is a Python script that implements the A* algorithm for path planning in a 2D grid map.
# Author: Hoang Pham
# Last modified: 2024-03-21
# Python version: 3.8
# Usage: python a_star_hoang_pham.py
# Notes: 

from queue import PriorityQueue
import numpy as np
import cv2

#############################################################################################

# Step 1: Define the Node class and actions

#############################################################################################
class Node:
    def __init__(self, state, parent=None, cost_to_come = 0, cost = float('inf')):
        self.state = state
        self.parent = parent
        self.cost_to_come = cost_to_come
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.state == other.state
        return False

    def __hash__(self):
        return hash(self.state)

# Define the action functions
def move_straight(node, step_size):
    x, y, theta = node.state
    new_x = x + step_size * np.cos(np.deg2rad(theta))
    new_y = y + step_size * np.sin(np.deg2rad(theta))
    new_theta = theta
    return (new_x, new_y, new_theta, 1)

def move_30_left(node, step_size):
    x, y, theta = node.state
    new_x = x + step_size * np.cos(np.deg2rad(theta + 30))
    new_y = y + step_size * np.sin(np.deg2rad(theta + 30))
    new_theta = theta + 30
    return (new_x, new_y, new_theta, 1)

def move_60_left(node, step_size):
    x, y, theta = node.state
    new_x = x + step_size * np.cos(np.deg2rad(theta + 60))
    new_y = y + step_size * np.sin(np.deg2rad(theta + 60))
    new_theta = theta + 60
    return (new_x, new_y, new_theta, 1)

def move_30_right(node, step_size):
    x, y, theta = node.state
    new_x = x + step_size * np.cos(np.deg2rad(theta - 30))
    new_y = y + step_size * np.sin(np.deg2rad(theta - 30))
    new_theta = theta - 30
    return (new_x, new_y, new_theta, 1)

def move_60_right(node, step_size):
    x, y, theta = node.state
    new_x = x + step_size * np.cos(np.deg2rad(theta - 60))
    new_y = y + step_size * np.sin(np.deg2rad(theta - 60))
    new_theta = theta - 60
    return (new_x, new_y, new_theta, 1)

#############################################################################################

# Step 2: Define the configuration space with obstacles using mathematical equations

#############################################################################################
def create_map(height=500, width=1200, border_thickness=5):
    # Pre-compute obstacle positions
    obstacle_positions = set()
    border_positions = set()

    # Add border obstacle
    for y in range(height):
        for x in range(width):
            if x < border_thickness or x >= width - border_thickness or y < border_thickness or y >= height - border_thickness:
                border_positions.add((x, y))

    # Rectangle obstacle 1
    for y in range(100 - 5, 500):
        for x in range(100 - 5, 175 + 5):
            if 100 <= x < 175 and 100 <= y < 500:
                obstacle_positions.add((x, y))
            else:
                border_positions.add((x, y))

    # Rectangle obstacle 2
    for y in range(0, 400 + 5):
        for x in range(275 - 5, 350 + 5):
            if 275 <= x < 350 and 0 <= y < 400:
                obstacle_positions.add((x, y))
            else:
                border_positions.add((x, y))

    # Polygonal obstacle with 6 sides (hexagon)
    for y in range(100 - 5, 400 + 6):
        for x in range(510 - 5, 790 + 6):
            if y - 0.615384615385 * x <= 5 and \
                y + 0.576923076923 * x - 775 <= 5 and \
                x <= 785 and \
                y - 0.576923076923 * x + 275 >= -5 and \
                y + 0.576923076923 * x - 475 >= -5 and \
                x >= 510:
                    obstacle_positions.add((x, y))

    # Obstacle made from rectangles on the far right
    for y in range(50-5, 125 + 5):
        for x in range(900 - 5, 1100 + 5):
            if 900 <= x < 1100 and 50 <= y < 125:
                obstacle_positions.add((x, y))
            else:
                border_positions.add((x, y))
    for y in range(375 - 5, 450 + 5):
        for x in range(900 - 5, 1100 + 5):
            if 900 <= x < 1100 and 375 <= y < 450:
                obstacle_positions.add((x, y))
            else:
                border_positions.add((x, y))
    for y in range(50 - 5, 450 + 5):
        for x in range(1020 - 5, 1100 + 5):
            if 1020 <= x < 1100 and 125 <= y < 450:
                obstacle_positions.add((x, y))
            else:
                border_positions.add((x, y))

    # Create an empty canvas
    canvas = np.zeros((height, width), dtype=np.uint8)

    # Draw obstacles on the canvas
    for y in range(height):
        for x in range(width):
            if (x, height - 1 - y) in obstacle_positions:
                canvas[y, x] = 255  # White represents obstacles
            elif (x, height - 1 - y) in border_positions:
                canvas[y, x] = 128  # Gray represents borders

    # Obstacles made from borders
    for coord in border_positions:
        obstacle_positions.add(coord)        

    return canvas, height, obstacle_positions

#############################################################################################

# Step 3: Implement the A* algorithm to generate the graph and find the optimal path

#############################################################################################
# Define the heuristic function
def euclidean_distance(state1, state2):
    x1, y1, theta1 = state1
    x2, y2, theta2 = state2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Define the goal node check function
def is_goal_node(current_node, goal_node, threshold=1.5):
    return euclidean_distance(current_node.state, goal_node.state) <= threshold

# Define the goal node check function
def is_duplicate_node(V, node, threshold_xy=0.5, threshold_theta=30):
    x, y, theta = node.state
    i = int(round(y / threshold_xy))
    j = int(round(x / threshold_xy))
    k = int(round(theta / threshold_theta))
    if V[i][j][k] == 1:
        return True
    else:
        V[i][j][k] = 1
        return False
    
# Define the A* algorithm
def a_star(start_node, goal_node, is_obstacle):
    open_list = PriorityQueue()
    
    threshold_xy = 0.5
    threshold_theta = 30
    V = np.zeros((500, 1200, 12), dtype=int)  # Matrix to store visited nodes information

    closed_list = set()
    
    start_node.cost_to_come = 0
    start_node.cost_to_go = euclidean_distance(start_node.state, goal_node.state)
    start_node.cost = start_node.cost_to_come + start_node.cost_to_go
    open_list.put(start_node)

    while not open_list.empty():
        current_node = open_list.get()

        closed_list.add(current_node)

        if is_goal_node(current_node, goal_node):
            return "Success", current_node, closed_list
        else:
            for action in actions:
                new_node = Node((0, 0, 0))
                x, y, theta, action_cost = action(current_node)
                new_node.state = (x, y, theta)

                if not is_duplicate_node(V, new_node) and not is_obstacle(new_node.state[0], new_node.state[1]):
                    if new_node.state not in [node.state for node in open_list.queue]:
                        new_node.parent = current_node
                        new_node.cost_to_come = current_node.cost_to_come + action_cost
                        new_node.cost_to_go = euclidean_distance(new_node.state, goal_node.state)
                        new_node.cost = new_node.cost_to_come + new_node.cost_to_go
                        open_list.put(new_node)
                else:
                    if new_node.cost > current_node.cost_to_come + action_cost + euclidean_distance(new_node.state, goal_node.state):
                        new_node.parent = current_node
                        new_node.cost_to_come = current_node.cost_to_come + action_cost
                        new_node.cost_to_go = euclidean_distance(new_node.state, goal_node.state)
                        new_node.cost = new_node.cost_to_come + new_node.cost_to_go

    return None

#############################################################################################

# Step 4: Implement the backtracking function to find the optimal path

#############################################################################################
# Define the backtracking function
def backtrack(goal_node):
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node)
        current_node = current_node.parent
    path.reverse()
    return path

#############################################################################################

# Step 5: Implement the visualization function to display the map, explored nodes, and the optimal path

#############################################################################################
def visualize_path(canvas, path_nodes, closed_list):
    # Create a canvas copy with three channels for RGB
    canvas_copy = np.zeros((canvas.shape[0], canvas.shape[1], 3), dtype=np.uint8)
    
    # Define colors in RGB format
    free_space_color = (0, 0, 0)  # Black for free space
    obstacle_color = (255, 255, 255)  # White for obstacles
    border_color = (0, 128, 255)  # Gray for borders
    explored_color = (255, 0, 255)  # Yellow for explored nodes
    path_color = (0, 255, 0)  # Green for path
    
    # Draw free space, obstacles, and borders onto the canvas
    for y in range(canvas.shape[0]):
        for x in range(canvas.shape[1]):
            if canvas[y, x] == 0:
                canvas_copy[y, x] = free_space_color
            elif canvas[y, x] == 255:
                canvas_copy[y, x] = obstacle_color
            elif canvas[y, x] == 128:
                canvas_copy[y, x] = border_color
    
    # Sort the closed list by the order of exploration
    closed_list_sorted = sorted(closed_list, key=lambda node: node.cost_to_come)
    
    # Define frame skip interval
    frame_skip_interval = 10
    frame_count = 0
    
    # Show the node exploration animation with frame skipping
    for node in closed_list_sorted:
        frame_count += 1
        if frame_count % frame_skip_interval != 0:
            continue
        x, y = node.state
        canvas_copy[height - 1 - y, x] = explored_color
        cv2.imshow("Path Planning", canvas_copy)
        cv2.waitKey(1)  # 1 ms delay between frames
    
    # Show the optimal path animation
    for node in path_nodes:
        x, y = node.state
        canvas_copy[height - 1 - y, x] = path_color
        cv2.imshow("Path Planning", canvas_copy)
        cv2.waitKey(10)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to get start and goal nodes from user input
def get_start_and_goal_nodes(obstacle_positions):
    while True:
        start_x, start_y = map(int, input("Enter the start node coordinates (x y): ").split())
        if (start_x, start_y) not in obstacle_positions:
            break
        else:
            print("Start node coordinates are within an obstacle. Please choose different coordinates.")

    while True:
        goal_x, goal_y = map(int, input("Enter the goal node coordinates (x y): ").split())
        if (goal_x, goal_y) not in obstacle_positions:
            break
        else:
            print("Goal node coordinates are within an obstacle. Please choose different coordinates.")

    start_node = Node((start_x, start_y))
    goal_node = Node((goal_x, goal_y))

    return start_node, goal_node

#############################################################################################

# Step 6: Main function to run the path planning algorithm

#############################################################################################

if __name__ == "__main__":
    
    # Define the action set
    actions = [
        move_straight,
        move_30_left,
        move_60_left,
        move_30_right,
        move_60_right
    ]

    # Create the map
    canvas, height, obstacle_positions = create_map()
    
    # Define the is_obstacle function using pre-computed obstacle positions
    def is_obstacle(x, y):
        return (x, y) in obstacle_positions
    
    ########################################### (Uncomment to preview the map using OpenCV)
    # cv2.imshow("Map", canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ###########################################

    # Get user input for start and goal nodes
    start_node, goal_node = get_start_and_goal_nodes(obstacle_positions)

    # if result == "Success":
    #     print("Path found!")
    #     path_nodes = backtrack(goal_node)
    #     visualize_path(canvas, path_nodes, closed_list)
    # else:
    #     print("No path found.")
