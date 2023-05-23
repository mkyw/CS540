from enum import unique
import heapq
import math
from os import stat
from turtle import distance


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0

    for i in range(9):
        if (from_state[i] == 0):
            continue
        from_row = math.floor(i/3)
        from_column = i % 3
        to_row = -1
        to_column = -1
        for j in range(9):
            if (to_state[j] == from_state[i]):
                to_row = math.floor(j/3)
                to_column = j % 3
        distance += abs(to_row - from_row) + abs(to_column - from_column)

    return distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []
    a = -1
    a_row = -1
    a_column = -1
    b = -1
    b_row = -1
    b_column = -1
    for i in range(9):
        if state[i] == 0:
            if a == -1:
                a = i
                a_row = math.floor(i/3)
                a_column = i % 3
            else:
                b = i
                b_row = math.floor(i/3)
                b_column = i % 3

    # Check up for a
    if ((a_row - 1 >= 0) and (state[a-3] != 0)):
        temp_state = state.copy()
        temp = temp_state[a]
        temp_state[a] = temp_state[a-3]
        temp_state[a-3] = temp
        succ_states.append(temp_state)

    # Check down for a
    if ((a_row + 1 <= 2) and (state[a+3] != 0)):
        temp_state = state.copy()
        temp = temp_state[a]
        temp_state[a] = temp_state[a+3]
        temp_state[a+3] = temp
        succ_states.append(temp_state)

    # Check left for a
    if ((a_column - 1 >= 0) and (state[a-1] != 0)):
        temp_state = state.copy()
        temp = temp_state[a]
        temp_state[a] = temp_state[a-1]
        temp_state[a-1] = temp
        succ_states.append(temp_state)

    # Check right for a
    if ((a_column + 1 <= 2) and (state[a+1] != 0)):
        temp_state = state.copy()
        temp = temp_state[a]
        temp_state[a] = temp_state[a+1]
        temp_state[a+1] = temp
        succ_states.append(temp_state)

    # Check up for b
    if ((b_row - 1 >= 0) and (state[b-3] != 0)):
        temp_state = state.copy()
        temp = temp_state[b]
        temp_state[b] = temp_state[b-3]
        temp_state[b-3] = temp
        succ_states.append(temp_state)

    # Check down for b
    if ((b_row + 1 <= 2) and (state[b+3] != 0)):
        temp_state = state.copy()
        temp = temp_state[b]
        temp_state[b] = temp_state[b+3]
        temp_state[b+3] = temp
        succ_states.append(temp_state)

    # Check left for b
    if ((b_column - 1 >= 0) and (state[b-1] != 0)):
        temp_state = state.copy()
        temp = temp_state[b]
        temp_state[b] = temp_state[b-1]
        temp_state[b-1] = temp
        succ_states.append(temp_state)

    # Check right for b
    if ((b_column + 1 <= 2) and (state[b+1] != 0)):
        temp_state = state.copy()
        temp = temp_state[b]
        temp_state[b] = temp_state[b+1]
        temp_state[b+1] = temp
        succ_states.append(temp_state)

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    open = []  # a priority queue
    max_len = 1

    g = 0
    h = get_manhattan_distance(state, goal_state)
    cost = g-h
    parent_id = -1
    unique_id = 0

    heapq.heappush(open, (h, state, (g, h, parent_id, unique_id)))
    visited = []
    visited_nodes = []
    while open:
        popped = heapq.heappop(open)
        c, s, other_infor = popped
        visited_nodes.append(popped)
        visited.append(s)

        if s == goal_state:
            solution = []
            current = popped
            while True:
                solution.insert(0, current[1])
                if (current[2][2] == -1):
                    break
                else:
                    for i in visited_nodes:
                        if (i[2][3] == current[2][2]):
                            current = i
            
            for i in range(len(solution)):
                print(str(solution[i]) + ' h='+ str(get_manhattan_distance(solution[i])) + ' moves: '+ str(i))
            print("Max queue length:", max_len)
            return

        succ_state = get_succ(s)
        for i in succ_state:
            if i not in visited:
                h = get_manhattan_distance(i, goal_state)
                g = other_infor[0]
                f = g + h
                unique_id += 1
                heapq.heappush(open, (f, i, (g + 1, h, other_infor[3], unique_id)))

            max_len = max(len(open), max_len)