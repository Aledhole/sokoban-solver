import operator
import functools


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                           UTILS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def find_1D_iterator(line, char):
    '''
    Return a generator that yield the positions (offset indices)
       where the character 'char' appears in the  'line' string.
    line : a string where we might find occurences of the 'char' character.
    char : a character that we are looking for.
    '''
    pos = 0
    # To see the doc of the string method 'find',  type  help(''.find)
    pos = line.find(char, pos)
    while pos != -1:
        yield pos
        pos = line.find(char, pos+1)


def find_2D_iterator(lines, char):
    '''
    Return a generator that  yields the (x,y) positions of
       the occurences of the character 'char' in the list of string 'lines'.
       A tuple (x,y) is returned, where
          x is the horizontal coord (column offset),
          and  y is the vertical coord (row offset)
    lines : a list of strings.
    char : the character we are looking for.
    '''
    for y, line in enumerate(lines):
        for x in find_1D_iterator(line, char):
            yield (x,y)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
class Warehouse:
    
    def copy(self, worker = None, boxes = None, weights = None):
        
        clone = Warehouse()
        clone.worker = worker or self.worker
        clone.boxes = boxes or self.boxes
        clone.weights = weights or self.weights
        clone.targets = self.targets
        clone.walls = self.walls
        clone.ncols = self.ncols
        clone.nrows = self.nrows
        return clone

    def from_string(self, warehouse_str):
        '''
        Create a warehouse from a string.
        '''
        lines = warehouse_str.split(sep='\n')
        self.from_lines(lines)

    def load_warehouse(self, filePath):
        '''
        Load a warehouse stored in a text file.
        '''
        with open(filePath, 'r') as f:
            # 'lines' is a list of strings (rows of the puzzle) 
            lines = f.readlines() 
        self.from_lines(lines)
        # self.weights = None # all boxes are of weight 1
            
    def from_lines(self, lines):        
        # Put the warehouse in a canonical format
        # where row 0 and column 0 have both at least one brick.
        first_row_brick, first_column_brick = None, None
        for row, line in enumerate(lines):
            brick_column = line.find('#')
            if brick_column>=0: 
                if  first_row_brick is None:
                    first_row_brick = row # found first row with a brick
                if first_column_brick is None:
                    first_column_brick = brick_column
                else:
                    first_column_brick = min(first_column_brick, brick_column)
        if first_row_brick is None:
            raise ValueError('Warehouse with no walls!')
        # compute the canonical representation
        # keep only the lines that contain walls
        canonical_lines = [line[first_column_brick:] 
                           for line in lines[first_row_brick:] if line.find('#')>=0]
        
        self.ncols = 1+max(line.rfind('#') for line in canonical_lines)
        self.nrows = len(canonical_lines)
        self.extract_locations(canonical_lines)
        
        # Weights if provided are on the first line   
        try:
            W = [int(v) for v in lines[0].split()]
        except:
            W = None
        # W can be an empty list if first line is blank
        if W is not None and len(W)>0:
            assert len(W) == len(self.boxes)
            self.weights = W
        else:
            self.weights = [0] * len(self.boxes)   
        #debug
        # print(f'{self.weights}')
    
    def save_warehouse(self, filePath):
       
        with open(filePath, 'w') as f:
            f.write(self.__str__())

    def extract_locations(self, lines):        
        workers =  list(find_2D_iterator(lines, "@"))  # workers on a free cell
        workers_on_a_target = list(find_2D_iterator(lines, "!"))
        # Check that we have exactly one agent
        assert len(workers)+len(workers_on_a_target) == 1 
        if len(workers) == 1:
            self.worker = workers[0]
        self.boxes = list(find_2D_iterator(lines, "$")) # crate/box
        self.targets = list(find_2D_iterator(lines, ".")) # empty target
        targets_with_boxes = list(find_2D_iterator(lines, "*")) # box on target
        self.boxes += targets_with_boxes
        # organize the boxes in reading order
        self.boxes.sort(key=lambda p:(p[1],p[0]))
        self.targets += targets_with_boxes
        if len(workers_on_a_target) == 1:
            self.worker = workers_on_a_target[0]
            self.targets.append(self.worker) 
        self.walls = list(find_2D_iterator(lines, "#")) # set(find_2D_iterator(lines, "#"))
        assert len(self.boxes) == len(self.targets)

    def __str__(self):       
        
        X,Y = zip(*self.walls) # pythonic version of the above
        x_size, y_size = 1+max(X), 1+max(Y)
        
        vis = [[" "] * x_size for y in range(y_size)]
        # can't use  vis = [" " * x_size for y ...]
        # because we want to change the characters later
        for (x,y) in self.walls:
            vis[y][x] = "#"
        for (x,y) in self.targets:
            vis[y][x] = "."
        # if worker is on a target display a "!", otherwise a "@"
        # exploit the fact that Targets has been already processed
        if vis[self.worker[1]][self.worker[0]] == ".": # Note y is worker[1], x is worker[0]
            vis[self.worker[1]][self.worker[0]] = "!"
        else:
            vis[self.worker[1]][self.worker[0]] = "@"
        # if a box is on a target display a "*"
        # exploit the fact that Targets has been already processed
        for (x,y) in self.boxes:
            if vis[y][x] == ".": # if on target
                vis[y][x] = "*"
            else:
                vis[y][x] = "$"
        return "\n".join(["".join(line) for line in vis])
    

    def __hash__(self):
        return hash(self.worker) ^ functools.reduce(operator.xor, [hash(box) for box in self.boxes])
    
if __name__ == "__main__":
    wh = Warehouse()
    wh.load_warehouse("./warehouses/warehouse_03.txt")

    print(wh)   # this calls    wh.__str__()


# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
#                              CODE CEMETARY
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +


