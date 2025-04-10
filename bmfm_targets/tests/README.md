# bmfm_targets.tests

There are a number of things new devs should know about the tests in this package.

## Unit tests

All of the basic package logic is covered by unit tests. Every new feature must have ample testing. Travis is currently set to break
if coverage drops below 70%.

## Dataset tests

New datasets that we want to support are expected to come with tests. This requires creating a small snippet of the data and adding it to
`tests/resources/` and adding a `test_new_dataset.py` file within `datasets`.

Each new file does not cover new package functionality, and if considered as unit tests, they are redundant. However, they validate that the supported
datasets are actually supported, and thus serve an important role.

For developers of the main package, it may be desirable to skip running those tests while developing other parts of the package because they can be slow.
This can be accomplished by setting the environment variable `BMFM_TARGETS_TESTS_SKIP_DATASET_TESTS` equal to a non-empty string.
If using VSCode, you can create a file called ".env" in your root directory and add the line `BMFM_TARGETS_TESTS_SKIP_DATASET_TESTS=1`.

## Integration tests

We have a number of end-to-end integration tests as well. These tests run an entire training loop using real data that is packaged in `tests/resources`, including
validation, checkpointing and verifying artifact creation.

## Speed tips

Because of the breadth of testing, it can get slow. There are a few ways to track and improve the runtime that we have used and recommend future devs to adopt/evolve:

- pytest fixtures. Some of the boilerplate is slow and can be shared across tests. Such functions should be defined as fixtures with a suitable scope. Fixtures to be used across multiple test files must be defined in `conftest.py` with `scope="session"`.
- Use small amounts of data in tests--especially tests with actual models. Sequence lengths, batches and hidden sizes can all be shrunk to the bare minimum for the purpose of testing.
- `pytest --durations=40` At the end of the Travis run you will see a list of the slowest tests. Use this to prioritize improvements.
- [pytest-profiling](https://pypi.org/project/pytest-profiling/) can be used to generate detailed runtime graphs via `pytest --profile-svg ~/bmfm_targets/tests/test_slow_test_file.py`. You may be surprised when you discover where your slow test is actually spending its time.
