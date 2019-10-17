# (c) Kevin Dunn, 2019. MIT License.

import pkg_resources
name = "process_improve"
#__version__ = VERSION = version = pkg_resources.get_distribution('process_improve').version

# Expose the API
from . structures import (c, gather, expand_grid, supplement)
from . models import (lm, summary)
from . plotting import (pareto_plot, contour_plot, predict_plot,
                       interaction_plot, tradeoff_table)
from . designs_factorial import (full_factorial)

