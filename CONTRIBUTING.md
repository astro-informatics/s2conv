# Contributing

Thank you for your interest in contributing to s2conv!
We welcome contributions of all forms, including bug reports, feature requests, documentation, tests and code.

## Reporting bugs or requesting new features

If you have a question, please first check the [documentation](https://astro-informatics.github.io/s2conv) and the [existing issues](https://github.com/astro-informatics/s2conv/issues).
If neither answers it, please [raise an issue](https://github.com/astro-informatics/s2conv/issues/new).

When reporting a bug, describe the behaviour you expected, what you observe instead, and provide enough information for someone else to reproduce the problem.
Ideally this is a [_minimal reproducible example_](https://en.wikipedia.org/wiki/Minimal_reproducible_example): a short script that reproduces the error and is as small and simple as possible.
Please include the versions of s2conv, s2fft, JAX and Python you are using.

## Setting up a development environment

Clone the repository and install the package in editable mode with the test and documentation dependencies:

```bash
git clone https://github.com/astro-informatics/s2conv.git
cd s2conv
pip install -e ".[tests,docs]"
```

The package version is determined from the git history by [setuptools-scm](https://setuptools-scm.readthedocs.io), so install from a clone of the repository rather than from an archive of the source files.

Two further sets of optional dependencies are available:

- `ssht` (pyssht and ducc0), required by the `"jax_ssht"` backend and by the tests marked `ssht`;
- `notebooks` (marimo and matplotlib), required to run the notebook in `notebooks/`.

## Running the tests

Run the test suite from the root of the repository with

```bash
pytest
```

Some tests are slow or need optional dependencies, and are marked accordingly.
You can deselect them with, for example,

```bash
pytest -m "not slow and not ssht"
```

New functionality should come with tests, and bug fixes with a test that fails without the fix.

## Code style and linting

The Python code in s2conv follows the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html), and we use [Ruff](https://docs.astral.sh/ruff/) to lint and autoformat it.
The Ruff configuration is in `pyproject.toml`.
The Marimo notebooks in `notebooks/` have their own format and are excluded from Ruff.

We use [pre-commit](https://pre-commit.com/) hooks to check that changes respect the formatting and linting rules.
You can install the hooks in your local repository by [installing pre-commit](https://pre-commit.com/#install) and running

```bash
pre-commit install
```

from the root of the repository.
The hooks then run on any staged changes when you commit.
If there are problems with the changes, these are reported in the terminal, and where possible they are fixed automatically, in which case the updated files need to be staged and committed again.
You can also check every file in the repository with `pre-commit run --all-files`.

## Building the documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org).
From the root of the repository, run

```bash
cd docs
make html
```

and open `docs/_build/html/index.html` in a browser.

## Proposing changes to the repository

We use a branch and pull request model.
Before opening a pull request that proposes substantial changes, for example a new feature or a change to the public interface, please first [raise an issue](https://github.com/astro-informatics/s2conv/issues/new) outlining the problem the changes would address, so that the problem and the proposed solution can be discussed before significant time is invested.

If you have not made an open-source contribution via a pull request before, you may find this [detailed guide](https://www.asmeurer.com/git-workflow/) by [asmeurer](https://github.com/asmeurer) helpful.
The main steps are as follows:

1. Create a branch with a descriptive name.
   Contributors without write access to the repository should first [fork the repository](https://github.com/astro-informatics/s2conv/fork) and create the branch in a local clone of their fork.
2. Make the proposed changes on the branch, giving each commit a descriptive commit message.
3. Push the branch to GitHub.
4. Create a [pull request](https://github.com/astro-informatics/s2conv/compare) against `main`, giving it a descriptive title and explaining what you are changing and why.
   If the pull request resolves a specific issue, use [keywords](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests) to link the issue.
5. Make sure that all the automated checks (tests, linting and documentation build) pass on the pull request.
6. Await a review by one of the maintainers and address any comments.
7. Once the checks pass and a maintainer has approved the changes, the pull request can be (squash) merged.

## Recognising contributions

We recognise every kind of contribution, not just code, following the [all-contributors](https://allcontributors.org) specification: the contributors are listed in the [Contributors section of the README](https://github.com/astro-informatics/s2conv#contributors-), with an emoji for each [type of contribution](https://allcontributors.org/docs/en/emoji-key), for example code, documentation, tests, bug reports, ideas, research or review.

To add someone, a maintainer comments on an issue or pull request with

```text
@all-contributors please add @<username> for <contributions>
```

for example `@all-contributors please add @jasonmcewen for code, doc`.
The [all-contributors bot](https://allcontributors.org/docs/en/bot/overview) then opens a pull request that updates `.all-contributorsrc` and the table in `README.md`.
If you think you or someone else should be listed, please say so in an issue or pull request.
