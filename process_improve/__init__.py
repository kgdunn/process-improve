import pkg_resources
name = "process_improve"
__version__ = VERSION = version = pkg_resources.get_distribution('process_improve').version
#__all__ = ["plotting", "models", "structures", "datasets", "simulations"]

from structures import (c, gather)
from models import (lm, summary)
from plotting import (pareto_plot, contour_plot)

