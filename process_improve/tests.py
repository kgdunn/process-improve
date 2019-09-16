import unittest

import pandas as pd

from structures import (gather, c)


class TestStructures(unittest.TestCase):
    """
    Test the data structures.
    """

    def setUp(self):
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0,  0]
        self.C = [4, 5, 6, 4, 6]

        self.y = [52, 74, 62, 80, 50, 65]

    def test_create(self):
        A1 = c(*(self.A))
        A2 = c(*(self.A), index=['lo', 'hi', 'lo', 'hi', 'cp', 'hi'])
        B = c(*(self.B), name='B')
        C1 = c(*(self.C), range=(4, 6))
        C2 = c(*(self.C), center=5, range=(4, 6))
        C3 = c(*(self.C), lo=4, hi=6)
        C4 = c(*(self.C), lo=4, hi=6, name = 'C4')
        C5 = c([4, 5,  6,  4,  6], lo=4, hi=6, name = 'C5')
        y = c(*(self.y), name='conversion')

        self.assertTrue(isinstance(A1, pd.Series))
        self.assertTrue(A1.shape==(6,))
        self.assertTrue(hasattr(A1, '_pi_index'))
        self.assertTrue(isinstance(A2.index, pd.Index))
        self.assertTrue(hasattr(A2, '_pi_index'))
        self.assertTrue(A2.name == 'Unnamed')

        with self.assertRaises(IndexError):
            A2 = c(*(self.A), index=['lo', 'hi', 'lo', 'hi', 'cp'])

        self.assertTrue(B.shape==(6,))
        self.assertTrue(B.name =='B')
        self.assertTrue(B._range is None)

        self.assertTrue(C1._range == (4, 6))
        self.assertTrue(C2._center == 5)
        self.assertTrue(C2._range == (4, 6))
        self.assertTrue(C3._lo == 4)
        self.assertTrue(C3._hi == 6)
        self.assertTrue(C4._lo == 4)
        self.assertTrue(C4._hi == 6)
        self.assertTrue(C5._hi == 6)
        self.assertTrue(C5.name == 'C5')



        self.assertTrue(len(y) ==6)
        self.assertTrue(y.name == 'conversion')



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