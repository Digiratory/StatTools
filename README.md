# StatTools

This library allows to create and process long-term dependent datasets.

![GitHub Release](https://img.shields.io/github/v/release/Digiratory/StatTools?link=https%3A%2F%2Fpypi.org%2Fproject%2FFluctuationAnalysisTools%2F)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Digiratory/StatTools/run-tests.yml?label=tests)
![GitHub License](https://img.shields.io/github/license/Digiratory/StatTools)
![PyPI - Downloads](https://img.shields.io/pypi/dm/fluctuationanalysistools?link=https%3A%2F%2Fpypi.org%2Fproject%2FFluctuationAnalysisTools%2F)

## Installation

You can install FluctuationAnalysisTools from [PyPI](https://pypi.org/project/FluctuationAnalysisTools/).

```bash
pip install FluctuationAnalysisTools
```

Or you can clone the repository and build it using the command

```bash
pip install .
```

## Basis usage

1. To create a simple dataset with given Hurst parameter:

```python
from StatTools.filters import FilteredArray

h = 0.8                 # choose Hurst parameter
total_vectors = 1000    # total number of vectors in output
vectors_length = 1440   # each vector's length 
t = 8                   # threads in use during computation

correlated_vectors = Filter(h, vectors_length).generate(n_vectors=total_vectors,
                                                        threads=t, progress_bar=True)
```

### Generators

1. Example of sequence generation based on the Hurst exponent.

```python
from StatTools.generators.hurst_generator import LBFBmGenerator
h = 0.8             # choose Hurst parameter
filter_len = 40     # length of the optimized filter
base = 1.2          # the basis for the filter optimization algorithm
target_len = 4000   # number of generation iterations

generator = LBFBmGenerator(h, filter_len, base)
trajectory = []
    for value in islice(generator, target_len):
        trajectory.append(value)
```

For more information and generator validation, see [lbfbm_generator.ipynb](/research/lbfbm_generator.ipynb).

### Fluctuational Analysis

1. Example of Detrended Fluctuational Analysis (DFA)

```python
from StatTools.generators.base_filter import Filter
from StatTools.analysis.dfa import DFA

h = 0.7 # choose Hurst parameter
length = 6000 # vector's length
target_std = 1.0
target_mean = 0.0

generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
trajectory = generator.generate(n_vectors=1)

actual_mean = np.mean(trajectory)
actual_std = np.std(trajectory, ddof=1)
actual_h = DFA(trajectory).find_h()
print(actual_h) # Should print a value close to 0.7
```

## Contributors

* [Alexandr Kuzmenko](https://github.com/alexandr-1k)
* [Aleksandr Sinitca](https://github.com/Sinitca-Aleksandr)
* [Asya Lyanova](https://github.com/pipipyau)
