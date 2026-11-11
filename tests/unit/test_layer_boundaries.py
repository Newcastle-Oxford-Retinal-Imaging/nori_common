"""Enforce the contracts/adapters implementation boundary."""

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "nori_common"


def _imports_module(tree: ast.AST, prefix: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == prefix or alias.name.startswith(prefix + "."):
                    found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == prefix or module.startswith(prefix + "."):
                found.append(module)
    return found


def _python_files(package: str) -> list[Path]:
    return sorted((SRC_ROOT / package).rglob("*.py"))


def test_contracts_do_not_import_adapters_or_fits_or_astropy():
    offenders = []
    for path in _python_files("contracts"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for prefix in ("nori_common.adapters", "nori_common.fits", "astropy"):
            if _imports_module(tree, prefix):
                offenders.append(path.relative_to(SRC_ROOT).as_posix())
                break
    assert not offenders, f"contracts must be a leaf layer: {offenders}"


def test_validation_does_not_import_adapters_or_fits_or_astropy():
    offenders = []
    for path in _python_files("validation"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for prefix in ("nori_common.adapters", "nori_common.fits", "astropy"):
            if _imports_module(tree, prefix):
                offenders.append(path.relative_to(SRC_ROOT).as_posix())
                break
    assert not offenders, f"validation must not depend on storage: {offenders}"
