import pandas as pd
import numpy as np
import pop_graph as popg
import community

# graphics stuff
%config InlineBackend.figure_format = 'png'
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,12

traits = pd.read_csv('../data/demo/nyc_example_coney.csv', index_col = 0)

# population weights matrix
pw = popg.popwmat(traits)

# co-occurrence matrix
co = popg.co_occurence_matrix(traits, pw)

# simplified co occurrence matrix - for visualization purposes
co_simp = popg.simplify_co_occurrence(co)

# full graph
G = popg.popGraph(affmat = co, labels = traits.columns)

# best partition of full graph
part = community.best_partition(G)

# simplified graph
G_simp = popg.popGraph(affmat = co_simp, labels = traits.columns)

# draw simplified graph
np.random.seed(909)
popg.drawPopGraph(G_simp, traits, part)
