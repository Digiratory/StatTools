# Changelog

## 1.7.1

* [GH-23](https://github.com/Digiratory/StatTools/issues/23) feat: add Kasdin generator. fix: change first arg in lfilter in LBFBm generator.
* [GH-15](https://github.com/Digiratory/StatTools/issues/15) feat&fix: LBFBm generator update: generate with input value and return an increment instead of the absolute value of the signal.
* [GH-25](https://github.com/Digiratory/StatTools/issues/25) feat: Detrended Fluctuation Analysis (DFA) for a nonequidistant dataset.

## 1.7.0

* [GH-5](https://github.com/Digiratory/StatTools/issues/5) feat: add LBFBm generator, that generates a sequence based on the Hurst exponent.
* [PR-8](https://github.com/Digiratory/StatTools/pull/8) refactor: rework filter-based generator.
* [PR-8](https://github.com/Digiratory/StatTools/pull/8) tests: add new tests for DFA and generators.
* [GH-10](https://github.com/Digiratory/StatTools/issues/10) build: enable wheel building with setuptools-scm.
* [GH-10](https://github.com/Digiratory/StatTools/issues/10) doc: enchance pyproject.toml with urls for repository, issues, and changelog.

## 1.6.1

* [PR-3](https://github.com/Digiratory/StatTools/pull/3) feat: add conventional FA

## 1.6.0

* [GH-1](https://github.com/Digiratory/StatTools/issues/1) Add argument `n_integral=1` in `StatTools.analysis.dpcca.dpcca` to provide possibility to control integretion in the beggining of the dpcca(dfa) analysis pipeline.
* fix: failure is processes == 1 and 1d array
* fix: remove normalization from dpcca processing

## 1.0.1 - 1.0.9

* Minor updates

## 1.1.0

* Added C-compiled modules
