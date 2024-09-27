Running the Application
1. Launching the Sokoban GUI
To run the graphical interface, execute the gui_sokoban.py file:

python gui_sokoban.py
The GUI allows you to:

Load a warehouse puzzle from the warehouses/ directory using the menu option (File > Load Warehouse).
Control the worker using the arrow keys to manually move boxes and solve the puzzle.
Solve the puzzle automatically using the solver by selecting Solve > Plan action sequence from the menu.

2. Solving a Puzzle Manually in the GUI
In the GUI:

Arrow keys: Move the worker (Left, Right, Up, Down).
'R' key: Reset the current puzzle to its initial state.
'S' key: Step through the solver's action sequence one move at a time.
'H' key: Display help information in the console.

3. Using the Automated Solver
The solver uses A search* to find the optimal solution for a weighted Sokoban puzzle. The solver can be invoked through the GUI:

Solve > Plan action sequence: Starts the solver and outputs the solution (if found) along with the total cost of the moves.
Solve > Play action sequence: Automatically plays the solution step-by-step on the GUI.
Key Functions

1. Sokoban Puzzle Solver (mySokobanSolver.py)
solve_weighted_sokoban(warehouse):
Solves a weighted Sokoban puzzle using A* search.
Returns a list of actions and the total cost of the solution.
check_elem_action_seq(warehouse, action_seq):

Checks if a given sequence of actions is legal in the context of the warehouse.
Returns the final state or Impossible if the sequence is illegal.
taboo_cells(warehouse):

Identifies taboo cells in the warehouse where boxes should not be pushed, as they would make the puzzle unsolvable.
2. Search Algorithms (search.py)
Breadth-First Search (BFS) and Depth-First Search (DFS) implementations for solving search-based problems.
A Search*: Uses a heuristic to guide the search toward the goal state more efficiently.
3. Warehouse Handling (sokoban.py)
Warehouse class:
Handles loading, storing, and visualizing Sokoban puzzle layouts.
Provides a string representation of the current state of the puzzle and utilities for manipulating the warehouse layout.
Testing the Solver
You can test the solver outside the GUI by running the following commands in the terminal:

Run the Solver:
python mySokobanSolver.py
This will load a sample warehouse, attempt to solve it, and print the solution (if found) and the total cost.

Adding New Warehouse Puzzles
To add a new Sokoban puzzle:

Create a new text file in the warehouses/ folder (e.g., warehouse_new.txt).
Format the file using the following characters:

# for walls
. for target locations
$ for boxes
@ for the worker

Example puzzle:

######
# .  #
# $$ #
# @  #
######

Load the new puzzle in the GUI or test it with the solver.

