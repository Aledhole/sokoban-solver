
'''

    Sokoban assignment


The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

You are NOT allowed to change the defined interfaces.
In other words, you must fully adhere to the specifications of the 
functions, their arguments and returned values.
Changing the interfacce of a function will likely result in a fail 
for the test of your code. This is not negotiable! 

You have to make sure that your code works with the files provided 
(search.py and sokoban.py) as your code will be tested 
with the original copies of these files. 

Last modified by 2022-03-27  by f.maire@qut.edu.au
- clarifiy some comments, rename some functions
  (and hopefully didn't introduce any bug!)

'''

import search
import sokoban
import itertools
import time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)

    '''
    return [(9681345, 'Callum', 'Bennie'),
            (10894772, 'Aled', 'Hole'),
            ()]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def manhattan_distance(p, q):
    """
        Compute the Manhattan distance between two points p and q.

        @param p: a tuple (x, y) representing a point
        @param q: a tuple (x, y) representing a point

        @return: the Manhattan distance between p and q

    """
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def euclidean_distance(p, q):
    """
        Compute the Euclidean distance between two points p and q.

        @param p: a tuple (x, y) representing a point
        @param q: a tuple (x, y) representing a point

        @return: the Euclidean distance between p and q

    """
    return ((p[0] - q[0])**2 + (p[1] - q[1])**2)**0.5


def is_corner(cell, walls):
    '''
        Check if a cell is a corner

        @param cell: a tuple (x, y) representing a cell
        @param walls: a set of (x, y) tuples representing the wall cells

        @return: True if cell is a corner, False otherwise
    '''
    # Candidate walls are walls within 2 cells of the cell
    candidate_walls = [
        wall for wall in walls if euclidean_distance(cell, wall) < 2]
    # Find walls that are left/right or up/down of the cell or diagonal
    left_right_walls = [wall for wall in candidate_walls if wall[1] == cell[1]]
    up_down_walls = [wall for wall in candidate_walls if wall[0] == cell[0]]
    diag_walls = [wall for wall in candidate_walls if wall !=
                  cell and wall not in left_right_walls and wall not in up_down_walls]

    # If there is a wall in each direction, the cell is a corner
    if left_right_walls and up_down_walls and diag_walls:
        return True


def graph_search(puzzle, frontier):
    '''
    Perform a graph search on a puzzle using a given frontier.

    @param puzzle: a SokobanPuzzle object
    @param frontier: a search.Frontier object

    @return: a set of states that were visited during the search
    '''
    # Perform graph search
    frontier.append(search.Node(puzzle.initial))
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node.state)
        frontier.extend(child for child in node.expand(puzzle)
                        if child.state not in explored
                        and child not in frontier)
    return explored
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A "taboo cell" is by definition
    a cell inside a warehouse such that whenever a box get pushed on such 
    a cell then the puzzle becomes unsolvable. 

    Cells outside the warehouse are not taboo. It is a fail to tag an 
    outside cell as taboo.

    When determining the taboo cells, you must ignore all the existing boxes, 
    only consider the walls and the target  cells.  
    Use only the following rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.

    @param warehouse: 
        a Warehouse object with the worker inside the warehouse

    @return
       A string representing the warehouse with only the wall cells marked with 
       a '#' and the taboo cells marked with a 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    # Search for all reachable cells
    puzzle = SokobanPuzzle(warehouse)
    # Ignore all existing boxes
    puzzle.initial = (warehouse.worker, ())

    # Perform graph search
    frontier = search.FIFOQueue()
    explored = graph_search(puzzle, frontier)

    # Reachable cells are all the cells the worker visited
    reachable = [worker for worker, boxes in explored]

    # Find reachable corners
    corners = [cell for cell in reachable if is_corner(
        cell, warehouse.walls) and cell not in puzzle.targets]

    # Find all pairs of corners
    corner_pairs = list(itertools.combinations(corners, 2))

    # Find taboo cells
    taboo_cells = []
    for pair in corner_pairs:
        # Find the axis of the line between the corners
        if pair[0][0] == pair[1][0]:
            axis = 0
        elif pair[0][1] == pair[1][1]:
            axis = 1
        else:
            continue

        # Find the length of the line between the corners
        if axis == 0:
            length = abs(pair[0][1] - pair[1][1])
        elif axis == 1:
            length = abs(pair[0][0] - pair[1][0])

        # Order the corners along the axis
        if axis == 0:
            pair = tuple(sorted(pair, key=lambda cell: cell[1]))
        elif axis == 1:
            pair = tuple(sorted(pair, key=lambda cell: cell[0]))

        # Find all cells between the corners
        cells_between = []
        for i in range(1, length):
            if axis == 0:
                cell = (pair[0][0], pair[0][1] + i)
            else:
                cell = (pair[0][0] + i, pair[0][1])

            cells_between.append(cell)

        # Check if any of the cells between the corners are targets or walls
        if any(cell in puzzle.walls for cell in cells_between) or any(cell in puzzle.targets for cell in cells_between):
            continue

        # Check if the line between the corners runs along a wall
        if axis == 0:
            # Check walls left and right of the line
            cells_left = [(cell[0] - 1, cell[1])
                          for cell in cells_between if cell[0] > 0]
            cells_right = [(cell[0] + 1, cell[1])
                           for cell in cells_between if cell[0] < puzzle.warehouse_dims[0] - 1]
            # If all cells left or all cells right are walls, the line runs along a wall
            if not (all(cell in puzzle.walls for cell in cells_left) or all(cell in puzzle.walls for cell in cells_right)):
                continue
        elif axis == 1:
            # Check walls above and below the line
            cells_above = [(cell[0], cell[1] - 1)
                           for cell in cells_between if cell[1] > 0]
            cells_below = [(cell[0], cell[1] + 1)
                           for cell in cells_between if cell[1] < puzzle.warehouse_dims[1] - 1]
            # If all cells above or all cells below are walls, the line runs along a wall
            if not (all(cell in puzzle.walls for cell in cells_above) or all(cell in puzzle.walls for cell in cells_below)):
                continue

        # Add cells between to taboo cells if passes all checks
        taboo_cells.extend(cells_between)

    # Add corners to taboo cells
    taboo_cells.extend(corners)

    # Create a string representation of the warehouse with taboo cells marked
    taboo_warehouse = str(warehouse.copy((), taboo_cells, warehouse.weights))
    # Replace the boxes with X's
    taboo_warehouse = taboo_warehouse.replace('$', 'X')
    # Replace workers and targets with spaces
    taboo_warehouse = taboo_warehouse.replace('*', ' ')
    taboo_warehouse = taboo_warehouse.replace('.', ' ')
    taboo_warehouse = taboo_warehouse.replace('@', ' ')

    print(taboo_warehouse)

    return taboo_warehouse

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    '''
    An instance of the class 'SokobanPuzzle' represents a Sokoban puzzle.
    An instance contains information about the walls, the targets, the boxes
    and the worker.

    Your implementation should be fully compatible with the search functions of 
    the provided module 'search.py'. 

    '''

    def __init__(self, warehouse):
        '''
        Create a new SokobanPuzzle instance.
        warehouse : an instance of the Warehouse class.

        Warehouse Attributes:
        warehouse_dims : number of columns and rows of the warehouse
        walls : a set of (x,y) tuples representing the wall cells
        targets : a set of (x,y) tuples representing the target cells
        weights : a set of (x,y) tuples representing the weight cells

        Warehouse Initial State:
        warehouse.worker : a tuple (x,y) representing the worker cell
        warehouse.boxes : a set of (x,y) tuples representing the box cells
        initial : a tuple (worker, boxes) representing the initial state

        Warehouse Goal:
        goal : a set of the warehouse.targets

        '''

        # Warehouse Attributes
        self.warehouse_dims = (warehouse.ncols, warehouse.nrows)
        self.walls = warehouse.walls
        self.targets = warehouse.targets
        self.weights = warehouse.weights

        # Initial State (Tuple of movable objects). Worker and boxes
        self.initial = (warehouse.worker, warehouse.boxes)
        self.initial = tuple([tuple(x) for x in self.initial])

        # Goal
        self.goal = set(warehouse.targets)

        # Visual State check
        # print('\nInitial:\n')
        # print(str(warehouse))
        # print('\nGoal:\n')
        # goal_print = warehouse.copy(self.goal[0], self.goal[1])
        # print(str(goal_print))

        # # Checking Actions and Results
        # print('\nInitial:\n')
        # print(self.initial)

        # print('\nActions:\n')
        # print(self.actions(self.initial))

        # print('\nResults:\n')
        # print(self.result(self.initial, 'Right'))

    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state.

        Notes:
        - The possible actions are: 'Up', 'Down', 'Left', 'Right'
        - An action is legal if the move is not into a wall and does not push two or more boxes at the same time.

        @param state: a tuple (worker, boxes) representing the current state

        @return
            A list of the legal actions that can be executed in the state.
                For example, ['Left', 'Down', 'Up', 'Right'] if all actions are legal.

        """

        # Unpack the state
        worker = state[0]
        boxes = state[1]

        # # Check if Up, Down, Left, Right is possible
        actions = []
        # UP: Check if there is no wall
        if (worker[0], worker[1]-1) not in self.walls:
            # Check if there is a box. If there is, check if there is a box behind it
            if (worker[0], worker[1]-1) in boxes:
                if (worker[0], worker[1]-2) not in self.walls and (worker[0], worker[1]-2) not in boxes:
                    actions.append('Up')
            else:
                actions.append('Up')
        # DOWN: Check if there is no wall
        if (worker[0], worker[1]+1) not in self.walls:
            # Check if there is a box. If there is, check if there is a box behind it
            if (worker[0], worker[1]+1) in boxes:
                if (worker[0], worker[1]+2) not in self.walls and (worker[0], worker[1]+2) not in boxes:
                    actions.append('Down')
            else:
                actions.append('Down')
        # LEFT: Check if there is no wall
        if (worker[0]-1, worker[1]) not in self.walls:
            # Check if there is a box. If there is, check if there is a box behind it
            if (worker[0]-1, worker[1]) in boxes:
                if (worker[0]-2, worker[1]) not in self.walls and (worker[0]-2, worker[1]) not in boxes:
                    actions.append('Left')
            else:
                actions.append('Left')
        # RIGHT: Check if there is no wall
        if (worker[0]+1, worker[1]) not in self.walls:
            # Check if there is a box. If there is, check if there is a box behind it
            if (worker[0]+1, worker[1]) in boxes:
                if (worker[0]+2, worker[1]) not in self.walls and (worker[0]+2, worker[1]) not in boxes:
                    actions.append('Right')
            else:
                actions.append('Right')

        return actions

    def result(self, state, action):
        """
        Return the state that results from executing a legal action in the given state.

        Notes:
        - If the action is into a box, the box is moved and the worker is moved
        - If there is no box, the worker is moved

        @param state: a tuple (worker, boxes) representing the current state

        @param action: a string representing the action to be executed

        @return
            A tuple (worker, boxes) representing the state resulting from executing the action in the current state.

        """

        # Unpack the state
        next_state = list(state)
        worker = list(state[0])
        boxes = list(state[1])

        # Up
        if action == 'Up':
            # If there is a box, move the box and the worker
            if (worker[0], worker[1]-1) in boxes:
                # Move the box
                boxes[boxes.index((worker[0], worker[1]-1))
                      ] = (worker[0], worker[1]-2)
                # Move the worker
                worker = (worker[0], worker[1]-1)
            # If there is no box, just move the worker
            else:
                worker = (worker[0], worker[1]-1)
        # Down
        elif action == 'Down':
            # If there is a box, move the box and the worker
            if (worker[0], worker[1]+1) in boxes:
                # Move the box
                boxes[boxes.index((worker[0], worker[1]+1))
                      ] = (worker[0], worker[1]+2)
                # Move the worker
                worker = (worker[0], worker[1]+1)
            # If there is no box, just move the worker
            else:
                worker = (worker[0], worker[1]+1)
        # Left
        elif action == 'Left':
            # If there is a box, move the box and the worker
            if (worker[0]-1, worker[1]) in boxes:
                # Move the box
                boxes[boxes.index((worker[0]-1, worker[1]))
                      ] = (worker[0]-2, worker[1])
                # Move the worker
                worker = (worker[0]-1, worker[1])
            # If there is no box, just move the worker
            else:
                worker = (worker[0]-1, worker[1])
        # Right
        elif action == 'Right':
            # If there is a box, move the box and the worker
            if (worker[0]+1, worker[1]) in boxes:
                # Move the box
                boxes[boxes.index((worker[0]+1, worker[1]))
                      ] = (worker[0]+2, worker[1])
                # Move the worker
                worker = (worker[0]+1, worker[1])
            # If there is no box, just move the worker
            else:
                worker = (worker[0]+1, worker[1])

        # Return the new state
        next_state[0] = tuple(worker)
        next_state[1] = tuple(boxes)
        return tuple(next_state)

    def return_solution(self, goal_node):
        """
            Shows solution represented by a specific goal node.
            For example, goal node could be obtained by calling 
                goal_node = breadth_first_tree_search(problem)

            The solution is a list of actions that lead from the initial state to the goal state.

            Note:
                - goal_node.path() returns a list of nodes from the root to the goal node
                - node.action returns the action that was applied to the parent to get the node

            @param goal_node: a node in a search tree that has a state that satisfies the goal function

            @return: a list of actions that lead from the initial state to the goal state
                or None if there is no solution
        """

        # If there is a solution
        if goal_node is not None:
            # Get the path
            path = goal_node.path()
            # Create a list to store the actions
            actions = []
            # Append the actions from each node in the goal path to the actions lists
            for node in path:
                if node.action is not None:
                    actions.append(node.action)
            return actions
        # If there is no solution
        else:
            return None
            print('No solution found.')

    def goal_test(self, state):
        """
            Return True if the state is a goal.

           Note:
            - The goal is to have all boxes in the warehouses target locations
            - The state is a tuple (worker, boxes) where boxes is a tuple of box locations
            - The goal is a set of target locations for the boxes

            @param state: a tuple (worker, boxes) representing the current state

            @return: True if the state is a goal, False otherwise
        """

        # Unpack the state and convert to set
        boxes = set(state[1])
        # Check if boxes are a subset of the goal
        return boxes.issubset(self.goal)

    def path_cost(self, c, state1, action, state2):
        """
            Return the cost of a path that arrives at state2 from
            state1.

            The path cost is the sum of:
                - the cost of the path from the initial state to state1
                - the cost of moving the worker and any boxes

            Note:
                - action is not used for this problem
                - state1 and state2 are tuples (worker, boxes)
                - weights are an attribute of the warehouse
                - the cost of moving the worker is 1
                - the cost of moving a box is the weight of the box

            @param c: the cost of the path from the initial state to state1
            @param state1: a tuple (worker, boxes) representing the current state
            @param action: a string representing the action to be executed
            @param state2: a tuple (worker, boxes) representing the state resulting from executing the action in state1

            @return: the cost of a path that arrives at state2 from state1
        """

        # Unpack States and attributes
        boxes1 = state1[1]
        boxes2 = state2[1]
        weights = self.weights

        # Check if the number of boxes is the same and if the weights are the same
        assert(len(boxes1) == len(boxes2))

        # Check if a box was moved
        if boxes1 != boxes2:
            # Find the box that was moved
            for i, (b1, b2) in enumerate(zip(boxes1, boxes2)):
                # If the box coordinates are different then a box was moved
                if b1 != b2:
                    # Add the cost of the box weight and the cost of the worker movement
                    return c + weights[i] + 1
        else:
            # Otherwise, return the cost of worker movement
            return c + 1

    def h(self, node):
        """
            Returns the heuristic value for a given node using the Manhattan distance function.

            The heuristic value is a optimistic estimate of the cost of the optimal path from the 
            state at the node to a goal state. This heuristic finds the Manhattan distance
            of each box to its closest target location and finds the cost of moving
            the boxes there, plus the cost of moving the worker to the closest box.

            Note:
                - the Manhattan distance is the sum of the differences in the x and y coordinates

            @param node: a node in a search tree. The node contains a state tuple (worker, boxes)

            @return: the heuristic value for the node
        """

        # Unpack the state, attributes and goal
        worker = node.state[0]
        boxes = list(node.state[1])
        weights = self.weights
        goal = self.goal

        # Make a dictionary of boxes and their weights
        box_weights = dict(zip(boxes, weights))

        # Initialize the heuristic value
        h_value = 0
        # For each box compute the cost of moving the box to the nearest goal
        for box in box_weights.keys():
            # If the box is already in a goal, don't bother computing the cost
            if box not in goal:
                # Compute the Manhattan distance from the box to the nearest goal
                box_distance = min(manhattan_distance(box, target)
                                   for target in goal)
                # Compute the cost of moving the box
                h_value += box_distance * (box_weights[box] + 1)

        # Compute the cost of moving the worker to the nearest box
        min_worker_box_distance = min(
            manhattan_distance(worker, box) for box in boxes)
        h_value += min_worker_box_distance

        return h_value
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def check_elem_action_seq(warehouse, action_seq):
    '''

    Determine if the sequence of actions listed in 'action_seq' is legal or not.

    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.

    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']

    @return
        The string 'Impossible', if one of the action was not valid.
           For example, if the agent tries to push two boxes at the same time,
                        or push a box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''

    # Create a Sokoban Puzzle
    puzzle = SokobanPuzzle(warehouse)
    # Initialize the current state
    state = puzzle.initial
    # For each action check if it is part of the available actions of the current state
    for action in action_seq:
        # If the actions is not a legal action, return 'Impossible'
        if action not in puzzle.actions(state):
            return 'Impossible'
        # Otherwise, update the state
        else:
            state = puzzle.result(state, action)

    # Return the state as a string
    worker = state[0]
    boxes = state[1]
    return str(warehouse.copy(worker, boxes, warehouse.weights))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def solve_weighted_sokoban(warehouse):
    '''
    This function analyses the given warehouse.
    It returns the two items. The first item is an action sequence solution. 
    The second item is the total cost of this action sequence.

    @param 
     warehouse: a valid Warehouse object

    @return

        If puzzle cannot be solved 
            return 'Impossible', None

        If a solution was found, 
            return S, C 
            where S is a list of actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
            C is the total cost of the action sequence C

    '''
    # Create a Sokoban Puzzle
    puzzle = SokobanPuzzle(warehouse)
    # Search for a solution
    sol_ts = search.astar_graph_search(puzzle)

    # If no solution was found return 'Impossible', None
    if sol_ts is None:
        print('No solution found.')
        return 'Impossible', None
    else:
        # Return the solutions action sequence
        sol_action_seq = puzzle.return_solution(sol_ts)

        # Check if the solution is legal and return the goal state
        goal_state = check_elem_action_seq(warehouse, sol_action_seq)

        # Calculate the cost of the solution
        cost = 0
        # Get the initial state
        state1 = puzzle.initial
        for action in sol_action_seq:
            # Get the state after applying the action
            state2 = puzzle.result(state1, action)
            # Calculate the cost of the action
            cost = puzzle.path_cost(cost, state1, action, state2)
            # Update the state
            state1 = state2

        return sol_action_seq, cost


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    print('- '*40)
    start = time.time()
    print(23*2.3)

    # Create a Sokoban puzzles
    wh = sokoban.Warehouse()
    # Test 01, , 8a, 09, 47, 81
    wh.load_warehouse("./warehouses/warehouse_71.txt")
    # Solve the puzzle
    sol_action_seq, cost = solve_weighted_sokoban(wh)
    print("Solution found with a cost of", cost)
    # Taboo Testing
    taboo_cells(wh)
    end = time.time()
    print("Analysis took", end - start, "seconds")
