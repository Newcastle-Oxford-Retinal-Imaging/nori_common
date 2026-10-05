"""Contract tests for the package boundary.

``nori_common`` is consumed by projects that do not have the private ``nori``
application installed, so it must never import from ``nori``.  This is the
executable form of that rule.
"""

import ast
from pathlib import Path

import nori_common

SOURCE_ROOT = Path(nori_common.__file__).parent


def test_package_imports() -> None:
    assert nori_common.__name__ == "nori_common"


def test_public_api_is_reexported() -> None:
    import nori_common.fits

    assert nori_common.FitsFile is nori_common.fits.FitsFile
    for name in nori_common.__all__:
        assert hasattr(nori_common, name), f"{name} missing from package root"


def test_source_never_imports_nori() -> None:
    offenders = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "nori" or name.startswith("nori.") for name in names):
                offenders.append(path.relative_to(SOURCE_ROOT).as_posix())

    assert not offenders, (
        f"nori_common must not import the private nori package: {sorted(offenders)}"
    )
