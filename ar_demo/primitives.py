import numpy as np


class Primitive():

    def __init__(self):
        self.vertices_asfloat32 = None

    def get_vertices(self):
        if self.vertices_asfloat32 is None:
            self.vertices_asfloat32 = np.float32(self.vertices)
        return self.vertices_asfloat32

    def get_edges(self):
        return self.edges


class Cube(Primitive):

    def __init__(self):
        super(Cube, self).__init__()

        self.vertices = [[1, -1, -1],
                         [1, 1, -1],
                         [-1, 1, -1],
                         [-1, -1, -1],
                         [1, -1, 1],
                         [1, 1, 1],
                         [-1, -1, 1],
                         [-1, 1, 1]
                         ]

        self.edges = [[0, 1],
                      [0, 3],
                      [0, 4],
                      [2, 1],
                      [2, 3],
                      [2, 7],
                      [6, 3],
                      [6, 4],
                      [6, 7],
                      [5, 1],
                      [5, 4],
                      [5, 7]
                      ]


class Triangle(Primitive):

    def __init__(self):
        super(Triangle, self).__init__()

        self.vertices = [[0, 0, 0],
                         [0, 1, 0],
                         [1, 1, 0],
                         [1, 0, 0],
                         [0.5, 0.5, 4]
                         ]

        self.edges = [(0, 1),
                      (1, 2),
                      (2, 3),
                      (3, 0),
                      (0, 4),
                      (1, 4),
                      (2, 4),
                      (3, 4)]
