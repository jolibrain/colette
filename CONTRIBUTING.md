## Contributing Guidelines

### Code Formatting

We use [ruff](https://github.com/charliermarsh/ruff) for linting and formatting our code. Before committing any changes, please ensure that your code is formatted correctly by running the appropriate commands.

We provide two targets in the `Makefile` to help with this process:

- `make style`: Automatically formats your code using ruff.
- `make lint`: Runs ruff to check for any style or linting errors.

#### Before Committing

Before making any commit, **you must run the `make style` target** to ensure your code is formatted correctly:

```bash
make style
```

This will automatically fix any formatting issues in your code.

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
