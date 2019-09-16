import unittest

from structures import (gather, c)


class TestStructures(unittest.TestCase):
    """
    Test the data structures.
    """

    def setUp(self):
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0,  0]
        self.y = [52, 74, 62, 80, 50, 65]

    def test_create(self):
        A = c(*(self.A))
        B = c(*(self.B))
        y = c(*(self.y), name='conversion')

    def test_gather(self):
        A = c(self.A)
        B = c(self.B)
        y = c(self.y, name='conversion')

        expt = gather(A=A, B=B, y=y)


class TestModels(unittest.TestCase):
    pass
    #popped_corn = lm("y ~ A + B + A*B", expt)
    #summary(popped_corn)


if __name__ == '__main__':
    unittest.main()