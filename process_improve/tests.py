import os
import unittest
import numpy as np
import pandas as pd

from structures import (gather, c)
from models import Model, summary
from plotting import (pareto_plot, contour_plot)

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
        C4 = c(*(self.C), lo=4, hi=6, name='C4')
        C5 = c([4, 5,  6,  4,  6], lo=4, hi=6, name='C5')
        C6 = c(*(self.C), lo=5, hi=6, name='C6')
        y = c(*(self.y), name='conversion')

        self.assertTrue(isinstance(A1, pd.Series))
        self.assertTrue(A1.shape == (6,))
        self.assertTrue(hasattr(A1, '_pi_index'))
        self.assertTrue(hasattr(A1, 'name'))
        self.assertTrue(A1.name == 'Unnamed')
        self.assertTrue(hasattr(A1, '_lo'))
        self.assertTrue(A1._lo == -1)
        self.assertTrue(hasattr(A1, '_hi'))
        self.assertTrue(A1._hi == +1)
        self.assertTrue(hasattr(A1, '_range'))
        self.assertTrue(A1._range[0] == -1)
        self.assertTrue(A1._range[1] == +1)
        self.assertTrue(hasattr(A1, '_center'))
        self.assertTrue(A1._center == 0)


        self.assertTrue(isinstance(A2.index, pd.Index))
        self.assertTrue(hasattr(A2, '_pi_index'))
        self.assertTrue(A2.name == 'Unnamed')

        with self.assertRaises(IndexError):
            A2 = c(*(self.A), index=['lo', 'hi', 'lo', 'hi', 'cp'])

        self.assertTrue(B.shape == (6,))
        self.assertTrue(B.name == 'B')


        self.assertTrue(C1._range == (4, 6))
        self.assertTrue(C2._center == 5)
        self.assertTrue(C2._range == (4, 6))
        self.assertTrue(C3._lo == 4)
        self.assertTrue(C3._hi == 6)
        self.assertTrue(C4._lo == 4)
        self.assertTrue(C4._hi == 6)
        self.assertTrue(C5._hi == 6)
        self.assertTrue(C5.name == 'C5')

        # User says the low is 5, but the minimum is actually different
        self.assertTrue(C6._lo == 5)
        self.assertTrue(C6._range == (5, 6))


        self.assertTrue(len(y) == 6)
        self.assertTrue(y.name == 'conversion')

    def test_column_math(self):
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0,  0]
        A = c(*(self.A))
        B = c(*(self.B), name="B")
        C1 = A * B
        C2 = B * A

        self.assertTrue(np.all(C1.values == [+1, -1, -1, +1, 0, 0]))
        self.assertTrue(np.all(C1.index == A.index))
        self.assertTrue(np.all(C2.values == [+1, -1, -1, +1, 0, 0]))
        self.assertTrue(np.all(C2.index == A.index))

    def test_gather(self):
        A = c(*(self.A))
        B = c(*(self.B))
        y = c(*(self.y), name='conversion')

        expt = gather(A=A, B=B, y=y)
        self.assertTrue(expt.shape == (6, 3))


class TestModels(unittest.TestCase):
    pass
    # popped_corn = lm("y ~ A + B + A*B", expt)
    # summary(popped_corn)


class Test_API_usage(unittest.TestCase):
    def setUp(self):
        full_path = os.path.join(os.path.split(__file__)[0],
                                 'media', 'test_doe1.csv')
        self.assertTrue(os.path.exists(full_path))
        self.df1 = pd.read_csv(full_path)
        self.df1.set_index('Run order', inplace=True)

    def test_case_1(self):
        pass
        index = self.df1.index
        C = c(self.df1['C'], lo = self.df1['C'].min(), hi = self.df1['C'].max(),
              index=index)
        M = c(self.df1['M'], levels = self.df1['M'].unique())
        V = c(self.df1['V'], lo = self.df1['V'].min(), hi=self.df1['V'].max())
        B = c(self.df1['B'], levels = self.df1['B'].unique())

        y = self.df1['y']

        expt = gather(C, M, V, B, y)
        self.assertTrue(C.index == M.index)

        model = Model("log10(y) ~ C*M*B*V", expt)
        #summary(model)
        #pareto_plot(model)
        #contour_plot(model, C, M)
        #contour_plot(model, "C", "M")
        #predict_plot(model)





if __name__ == '__main__':
    unittest.main()
