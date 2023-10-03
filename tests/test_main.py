"""Test cases for the __main__ module."""
import pytest
from click.testing import CliRunner

from spapros.__main__ import spapros_cli


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(spapros_cli)
    assert result.exit_code == 0
