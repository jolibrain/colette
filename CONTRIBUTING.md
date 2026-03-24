## Contributing Guidelines

### CI Matrix And Merge Rules

We split CI into fast required checks and deeper optional coverage:

- Required for PR merge: Jenkins `PR Smoke` lane (`make ci-smoke`)
- Optional protected PR check: Jenkins `integration-stable` lane (`make ci-integration-stable`)
- Nightly/manual deep validation: Jenkins GPU matrix lane (`make ci-integration`, `make ci-pipeline-integration`, `make ci-e2e`)

Recommended branch protection setup:

- Require status check from Jenkins for PR smoke
- Do not require optional integration-stable status
- Do not require nightly GPU matrix statuses

This gives fast PR feedback while keeping realistic deeper GPU validation available.

### Code Formatting

We use [ruff](https://github.com/charliermarsh/ruff) for linting and formatting our code. Before committing any changes, please ensure that your code is formatted correctly by running the appropriate commands.

We provide two targets in the `Makefile` to help with this process:

- `make style`: Automatically formats your code using ruff.
- `make lint`: Runs ruff to check for any style or linting errors.
- `make test-smoke`: Runs the validated localhost-safe smoke baseline using the repo virtualenv when available.
- `make test-coverage`: Runs smoke baseline with coverage report and threshold enforcement.

#### Before Committing

Before making any commit, **you must run the `make style` target** to ensure your code is formatted correctly:

```bash
make style
```

This will automatically fix any formatting issues in your code.

To run the baseline smoke suite locally:

```bash
make test-smoke
```

To run smoke baseline with coverage reporting:

```bash
make test-coverage
```

By default coverage is checked against `src/colette` with a minimum threshold of `35%`.
You can override this locally, for example:

```bash
make test-coverage COV_MIN=30
```

#### Using a Pre-commit Hook
To make the formatting process easier and more consistent, we recommend using a pre-commit hook. This ensures that your code is formatted before every commit, reducing the chance of missing any formatting issues.

You can set up a pre-commit hook using the configuration below. Add this to your .pre-commit-config.yaml file in the root of your repository:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  # Ruff version.
  rev: v0.6.3
  hooks:
    # Run the linter.
    - id: ruff
      args: [ --fix ]
    # Run the formatter.
    - id: ruff-format
```

After adding this configuration, install the pre-commit hooks by running:

```bash
pre-commit install
```

Now, every time you make a commit, the code will automatically be formatted according to our style guide.
