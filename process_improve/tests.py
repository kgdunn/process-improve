import os
import unittest
import numpy as np
import pandas as pd

from structures import (gather, c, create_names)
from models import (lm, summary)
from plotting import (pareto_plot, contour_plot)



class TestStructures(unittest.TestCase):
    """
    Test the data structures.
    """

    def setUp(self):
        self.A = [-1, +1, -1, +1, 0, +1]
        self.B = [-1, -1, +1, +1, 0,  0]
        self.C = [4, 5, 6, 4, 6]
        self.D = [0, 1, 'green']
        self.y = [52, 74, 62, 80, 50, 65]

    def test_create_names(self):
        """ Creating factor names."
        """

        self.assertListEqual(create_names(5),  ["A", "B", "C", "D", "E"])

        self.assertListEqual(create_names(3, letters=False),
                             ["X1", "X2", "X3"])

        self.assertListEqual(create_names(3, letters=False, prefix='Q',
                                          start_at=9, padded=True),
                             ["Q09", "Q10", "Q11"])

        self.assertListEqual(create_names(4, letters=False, prefix='Z',
                                          start_at=99, padded=True),
                             ["Z099", "Z100", "Z101", "Z102"])



    def test_create_factors(self):
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
        self.assertTrue(hasattr(A1, '_pi_lo'))
        self.assertTrue(A1._pi_lo == -1)
        self.assertTrue(hasattr(A1, '_pi_hi'))
        self.assertTrue(A1._pi_hi == +1)
        self.assertTrue(hasattr(A1, '_pi_range'))
        self.assertTrue(A1._pi_range[0] == -1)
        self.assertTrue(A1._pi_range[1] == +1)
        self.assertTrue(hasattr(A1, '_pi_center'))
        self.assertTrue(A1._pi_center == 0)


        self.assertTrue(isinstance(A2.index, pd.Index))
        self.assertTrue(hasattr(A2, '_pi_index'))
        self.assertTrue(A2.name == 'Unnamed')

        with self.assertRaises(IndexError):
            A2 = c(*(self.A), index=['lo', 'hi', 'lo', 'hi', 'cp'])

        self.assertTrue(B.shape == (6,))
        self.assertTrue(B.name == 'B')


        self.assertTrue(C1._pi_range == (4, 6))
        self.assertTrue(C2._pi_center == 5)
        self.assertTrue(C2._pi_range == (4, 6))
        self.assertTrue(C3._pi_lo == 4)
        self.assertTrue(C3._pi_hi == 6)
        self.assertTrue(C4._pi_lo == 4)
        self.assertTrue(C4._pi_hi == 6)
        self.assertTrue(C5._pi_hi == 6)
        self.assertTrue(C5.name == 'C5')

        # User says the low is 5, but the minimum is actually different
        self.assertTrue(C6._pi_lo == 5)
        self.assertTrue(C6._pi_range == (5, 6))


        D = c(*(self.D))
        self.assertTrue(D._pi_numeric == True)


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

        expt = gather(A=A, B=B, y=y, title='Testing expt name')

        #Eventually this method must go to the "DF" class; currently in the
        # model class; not really appropriate there.
        self.assertTrue(expt.get_title() == 'Testing expt name')

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
        C = c(self.df1['C'],
              lo = self.df1['C'].min(),
              hi = self.df1['C'].max(),
              index=index, name='C')
        M = c(self.df1['M'], levels = self.df1['M'].unique(), name='M')
        V = c(self.df1['V'], lo = self.df1['V'].min(), hi=self.df1['V'].max(),
               name='V')
        B = c(self.df1['B'], name='B')
        self.assertTrue(B._pi_levels[B.name] == ['Ard', 'Eme'])


        y = self.df1['y']

        expt = gather(C=C, M=M, V=V, B=B, y=y)
        self.assertTrue(np.all(C.index == M.index))

        model = lm("np.log10(y) ~ C*M*B*V", expt)
        #summary(model)
        #pareto_plot(model)
        #contour_plot(model, C, M)
        #contour_plot(model, "C", "M")
        #predict_plot(model)

if __name__ == '__main__':
    unittest.main()
