import attr
from typing import List, Tuple
import numpy as np
from ortools.linear_solver import pywraplp


@attr.s(auto_attribs=True)
class FunkyGraph:
    nodes_n: int
    edges: List[Tuple[int, int, int]]
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
        B[i, eix] = 1
        B[j, eix] = 1
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
        for x in edge_variables:
            print(f'Edge {x.name()} traversed {x.solution_value()} times')
    else:
        print('The problem does not have an optimal solution.')

    print("     ")
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


if __name__ == '__main__':
    solve_traversal(PETERS)
