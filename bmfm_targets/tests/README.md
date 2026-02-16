# bmfm_targets.tests

## Unit tests

All of the basic package logic is covered by unit tests. Every new feature must have the appropriate unit tests.

## Integration tests

We have a number of end-to-end integration tests as well. These tests run an entire training loop using synthetic resampling of real data that is packaged in `tests/resources`, including
validation, checkpointing and verifying artifact creation. This resampling process identifies the data state by differentiating between raw counts and log-normalized data, modeling them via Negative Binomial and Gamma distributions respectively while adjusting for sparsity to match the original density. By aggregating rare populations and synthesizing new data based on these group-specific profiles, the process preserves the functional identity and cluster structure of the original dataset.


## Speed tips

Because of the breadth of testing, it can get slow. There are a few ways to track and improve the runtime that we have used and recommend future devs to adopt/evolve:

- pytest fixtures. Some of the boilerplate is slow and can be shared across tests. Such functions should be defined as fixtures with a suitable scope. Fixtures to be used across multiple test files must be defined in `conftest.py` with `scope="session"`.
- Use small amounts of data in tests--especially tests with actual models. Sequence lengths, batches and hidden sizes can all be shrunk to the bare minimum for the purpose of testing.
- `pytest --durations=40` At the end of the CI run you will see a list of the slowest tests. Use this to prioritize improvements.
- [pytest-profiling](https://pypi.org/project/pytest-profiling/) can be used to generate detailed runtime graphs via `pytest --profile-svg ~/bmfm_targets/tests/test_slow_test_file.py`.
