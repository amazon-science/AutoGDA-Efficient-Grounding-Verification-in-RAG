# Factual-Consistency-Summer-Project

This is a [PeruPython](https://builderhub.corp.amazon.com/docs/brazil/user-guide/python-peru.html)
project using `poetry` to build, test, and packate the application

**You will need to have your runtimes set up via `asdf` to match your build platform to
use `brazil-build` similarly to your build platform**.

## Choosing the python versions to build and test against

The version of Python that poetry expects is 3.10 and is defined in `pyproject.toml`

## Building

The build logic runs the `build.sh` is the starting point for any `brazil-build` which is relatively short
and straightforward enough that we recommend you read it directly to update/modify your build process.

## Contributing
All development by individuals should be done in their separate branches work and [code reviews](https://builderhub.corp.amazon.com/docs/crux/user-guide/howto.html) should be sent before merging them into mainline.

## Adding Dependencies

All of the testing and development dependencies are in the `pyproject.toml` and are meant to be managed using `poetry` commands. If you need to add a library built and published from source within your version set, you still add it as an idiomatic dependency but also add the package to the `build_after` clause of your `brazil.ion` to guarantee your package is rebuilt when the dependency is updated.

## Testing

`brazil-build` will build and run `poetry run pytest`.

## Vending/publishing

This library publishes to the Workspace repository will need to be included/installed as an idiomatic dependency (amzn_harpo_hallucination_detection) by packages and applications that consume it.