# StatTools
This library allows to create and process long-term dependent datasets.

## Installation:
Уou can clone the repository and build it using the command
```
pip install .
```

You can install FluctuationAnalysisTools from [PyPI](https://pypi.org/project/FluctuationAnalysisTools/).

```
pip install FluctuationAnalysisTools
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
from StatTools.generators.hurst_generator import HurstGenerator
h = 0.8             # choose Hurst parameter
filter_len = 40     # length of the optimized filter
base = 1.2          # the basis for the filter optimization algorithm
target_len = 4000   # number of generation iterations

generator = HurstGenerator(h, filter_len, base)
trajectory = []
    for value in islice(generator, target_len):
        trajectory.append(value)
```
For more information and generator validation, see [hurst-generator.ipynb](/research/hurst-generator.ipynb)

## Contributors

* [Alexandr Kuzmenko](https://github.com/alexandr-1k)
* [Aleksandr Sinitca](https://github.com/Sinitca-Aleksandr)
* [Asya Lyanova](https://github.com/pipipyau)
