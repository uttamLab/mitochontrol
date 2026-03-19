"""Smoke tests: verify that the package and all public symbols import."""


def test_import_package():
    import mitochontrol  # noqa: F401


def test_version_string():
    import mitochontrol

    assert isinstance(mitochontrol.__version__, str)
    parts = mitochontrol.__version__.split(".")
    assert len(parts) == 3


def test_all_symbols_importable():
    import mitochontrol

    for name in mitochontrol.__all__:
        assert hasattr(mitochontrol, name), (
            f"{name} is listed in __all__ but not importable"
        )
