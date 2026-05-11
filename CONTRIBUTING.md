# Contributing to signal-curator

Bug reports, feature requests, dataset-adapter requests, and pull requests
are all welcome. For substantial changes please open an issue first so we
can align on scope before you spend time coding.

## Setting up a dev environment

```bash
git clone https://github.com/johnstamly/signal-curator
cd signal-curator
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest tests/ -v
```

CI runs the same suite on Linux / macOS / Windows across Python 3.10-3.12.

## Linting

```bash
ruff check src/ scripts/ tests/
```

`ruff` config is in `pyproject.toml` (line length 100, target Python 3.10+).

## Adding a new dataset adapter

1. Read [`docs/ADAPTING.md`](docs/ADAPTING.md) -- decide if you actually need
   a new adapter or if the `GenericAdapter` works with pre-conversion
2. Implement the `DatasetAdapter` protocol from `signal_curator/dataset.py`
3. Add a test in `tests/test_<your_adapter>.py` that loads a small fixture
4. Add a one-line factory branch in `dataset.py::make_adapter`
5. Document the schema in `docs/`

## Commit / PR conventions

- One topic per PR -- keeps reviews fast
- Tests must pass on CI before merge
- Reference issues in the PR description (`closes #N`, `re #N`)
- For UI changes, please include a screenshot or short screencap

## License

By contributing you agree your changes are released under the MIT License.
