"""The GZIP level codec must stay contained within a single write.

Registering a custom codec means mutating astropy's process-global
``ALGORITHMS`` registry.  Every entry point must restore the original codec
afterwards so other astropy users in the same process are unaffected.
"""

import numpy as np
import pytest
from astropy.io.fits.hdu.compressed import _tiled_compression

from nori_common._gzip import gzip_level
from nori_common.fits import Compression, WritePolicy, write_fits

GZIP_KEY = "GZIP_1"


def _data():
    return np.arange(1000, dtype=np.float64).reshape(10, 100)


def test_gzip_level_context_installs_and_restores():
    original = _tiled_compression.ALGORITHMS[GZIP_KEY]
    with gzip_level(3):
        installed = _tiled_compression.ALGORITHMS[GZIP_KEY]
        assert installed is not original
        assert issubclass(installed, original)
    assert _tiled_compression.ALGORITHMS[GZIP_KEY] is original


def test_codec_is_restored_after_gzip_write(tmp_path):
    original = _tiled_compression.ALGORITHMS[GZIP_KEY]
    write_fits(
        tmp_path / "a.fits",
        _data(),
        policy=WritePolicy(compression=Compression.GZIP, gzip_level=1),
    )
    assert _tiled_compression.ALGORITHMS[GZIP_KEY] is original


def test_codec_is_restored_after_uncompressed_write(tmp_path):
    original = _tiled_compression.ALGORITHMS[GZIP_KEY]
    write_fits(
        tmp_path / "a.fits",
        _data(),
        policy=WritePolicy(compression=Compression.NONE),
    )
    assert _tiled_compression.ALGORITHMS[GZIP_KEY] is original


def test_codec_is_restored_when_write_fails(tmp_path):
    original = _tiled_compression.ALGORITHMS[GZIP_KEY]
    with pytest.raises(OSError):
        write_fits(
            tmp_path / "missing" / "a.fits",
            _data(),
            policy=WritePolicy(compression=Compression.GZIP),
        )
    assert _tiled_compression.ALGORITHMS[GZIP_KEY] is original


def test_gzip_level_affects_encoded_bytes(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, (300, 300)).astype(np.float64)
    low = tmp_path / "l1.fits"
    high = tmp_path / "l9.fits"
    write_fits(
        low,
        data,
        policy=WritePolicy(compression=Compression.GZIP, gzip_level=1),
    )
    write_fits(
        high,
        data,
        policy=WritePolicy(compression=Compression.GZIP, gzip_level=9),
    )
    assert low.read_bytes() != high.read_bytes()
