"""Nox sessions."""
import nox

nox.options.sessions = "lint", "tests"
locations = "src",

@nox.session(python=["3.8"])
def tests(session):
    """Run the test suite."""
    args = session.posargs or ["--cov=src"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)

@nox.session(python=["3.8"])
def lint(session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)

package = "random_radon_transformation"
@nox.session(python=["3"])
def xdoctest(session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.run("poetry", "install", "--no-dev", external=True)
    session.install("xdoctest")
    session.run("python", "-m", "xdoctest", package, *args)