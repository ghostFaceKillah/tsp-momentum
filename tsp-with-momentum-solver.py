import attr
import copy
from typing import List, Tuple, Dict
import numpy as np
from brainos.utils.result import Result
from ortools.linear_solver import pywraplp



Edge = Tuple[int, int, int]
Routing = Dict[Edge, List[Edge]]

@attr.s(auto_attribs=True)
class FunkyGraph:
    nodes_n: int
    edges: List[Edge]
    costs: List[float]


EASY = FunkyGraph(
    nodes_n=3,
    edges = [
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1)
    ],
    costs = [
        0.7,
        0.4,
        0.1
    ]
)

EASY_2 = FunkyGraph(
    nodes_n=4,
    edges=[
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 0),
        (3, 0, 1),
        (2, 0, 2),
        (0, 2, 0),
    ],
    costs = [
        1.0,
        0.4,
        0.4,
        0.4,
        0.4,
    ]
)

PETERS = FunkyGraph(
    nodes_n=8,
    edges=[
        (3, 0, 1),
        (5, 0, 1),
        (1, 0, 3),
        (1, 0, 5),

        (0, 1, 2),
        (0, 1, 7),
        (2, 1, 0),
        (7, 1, 0),

        (1, 2, 3),
        (1, 2, 4),
        (3, 2, 1),
        (4, 2, 1),

        (0, 3, 2),
        (0, 3, 4),
        (2, 3, 0),
        (4, 3, 0),

        (2, 4, 6),
        (3, 4, 6),
        (6, 4, 2),
        (6, 4, 3),

        (0, 5, 6),
        (0, 5, 7),
        (6, 5, 0),
        (7, 5, 0),

        (4, 6, 7),
        (5, 6, 7),
        (7, 6, 4),
        (7, 6, 5),

        (1, 7, 5),
        (1, 7, 6),
        (5, 7, 1),
        (6, 7, 1),

        (6, 7, 6),  # need to add non-smooth edge, otherwise the example graph is not traversable
        (7, 6, 7),
    ],
    costs=[1.0] * (4 * 8) + [100.0, 100.0]
)


def to_linear_address(i, j, nodes_n):
    assert i < nodes_n
    assert j < nodes_n
    return i * nodes_n + j


def to_edges_constraint_matrix(edges, nodes_n):
    A = np.zeros(shape=(nodes_n * nodes_n, len(edges)), dtype=np.int32)
    for eix, edge in enumerate(edges):
        i, j, k = edge
        A[to_linear_address(i, j, nodes_n), eix] = 1
        A[to_linear_address(j, k, nodes_n), eix] = -1

    nonzero_index = np.abs(A).sum(axis=1) > 1
    sparser = A[nonzero_index, :]
    return sparser


def to_visits_constaint_matrix(edges, nodes_n):
    B = np.zeros(shape=(nodes_n, len(edges)), dtype=np.int32)
    for eix, edge in enumerate(edges):
        i, j, k = edge
        B[j, eix] = 1
        B[i, eix] = 1
        B[k, eix] = 1

    return B


def solve_traversal(graph: FunkyGraph):
    solver = pywraplp.Solver.CreateSolver('SCIP')

    infinity = solver.infinity()
    # x and y are integer non-negative variables.

    edge_variables = []
    for edge in graph.edges:
        i, j, k = edge
        e = solver.IntVar(0.0, infinity, f'e_{i}{j}{k}')
        edge_variables.append(e)

    A = to_edges_constraint_matrix(graph.edges, graph.nodes_n)
    B = to_visits_constaint_matrix(graph.edges, graph.nodes_n)  # can assert visitability here

    print("edges constraint matrix")
    print(A)
    print("visits constraint matrix")
    print(B)

    for row in A:
        local_expression = 0
        for coeff, edge_var in zip(row, edge_variables):
            local_expression += coeff * edge_var
        solver.Add(local_expression == 0)

    for row in B:
        one_idx = np.where(row == 1)[0][0]
        solver.Add(edge_variables[one_idx] >= 1.0)

    cost = 0
    for cost_coeff, edge_var in zip(graph.costs, edge_variables):
        cost += cost_coeff * edge_var

    solver.Maximize(-cost)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        solution = []
        for x, edge in zip(edge_variables, graph.edges):
            for _ in range(int(x.solution_value())):
                solution.append(edge)
            print(f'Edge {x.name()} traversed {x.solution_value()} times')
    else:
        solution = Result.error("Didn't find solution")
        print('The problem does not have an optimal solution.')

    print("     ")
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

    return solution


# solution extraction

def add_to_data(from_edge: Edge, to_edge: Edge, data: Routing):
    if from_edge in data:
        data[from_edge].append(to_edge)
    else:
        data[from_edge] = [to_edge]


def remove_from_data(from_edge: Edge, to_edge: Edge, data: Routing):
    assert from_edge in data
    data[from_edge].remove(to_edge)


def is_empty(data: Routing):
    return all([len(to_edges) == 0 for to_edges in data.values()])


def almost_done(data: Routing):
    return sum([len(to_edges) for to_edges in data.values()]) == 1


def resolve_solution(start_from: int, from_to: Routing) -> List[int]:
    beginning_choices: List[Edge] = [(i, j) for i, j in from_to.keys() if i == start_from]

    assert len(beginning_choices) >= 1

    def traverse_edge(current_edge: Edge, from_to: Routing, solution: List[Edge]) -> Tuple[bool, List[Edge]]:
        if len(from_to[current_edge]) == 0:
            return False, solution
        elif almost_done(from_to):
            next_edge = from_to[current_edge][0]
            solution.append(next_edge)
            return True, solution
        else:
            possible_next_edges = copy.deepcopy(from_to[current_edge])
            for possible_next_edge in possible_next_edges:
                remove_from_data(current_edge, possible_next_edge, from_to)
                solution.append(possible_next_edge)

                success, solution = traverse_edge(possible_next_edge, from_to, solution)
                if success:
                    return True, solution

                # add edge back to the data
                add_to_data(current_edge, possible_next_edge, from_to)
                # remove the edge from current solution candidate
                solution.pop()

            return False, solution

    solution_so_far = []
    for start_edge in beginning_choices:
        solution_so_far.append(start_edge)
        success, solution = traverse_edge(start_edge, from_to, solution_so_far)
        if success:
            break
        else:
            solution_so_far.pop()

    if success:
        # unpack solution
        solution_nodes = []
        for i, j in solution:
            solution_nodes.append(i)
        return solution_nodes
    else:
        raise AssertionError("Failed to find solution")
    pass


if __name__ == '__main__':
    graph = PETERS
    solution = solve_traversal(graph)
    print(solution)

    from_to = {}
    for i, j, k in solution:
        add_to_data(from_edge=(i, j), to_edge=(j, k), data=from_to)

    for start_from in range(graph.nodes_n):
        solution = resolve_solution(start_from, copy.deepcopy(from_to))
        print(f'Solution starting from {start_from} is {solution}')
