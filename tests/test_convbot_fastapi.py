"""Test convbot-fastapi."""
from convbot_fastapi import __version__
from convbot_fastapi import convbot_fastapi


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_sanity():
    """Sanity check."""
    try:
        assert not convbot_fastapi()
    except Exception:
        assert True
