"""Regression checks for Kerchunk reference handling (GHSA-27vj-qcqg-25rc)."""

import json
from importlib.metadata import version

import fsspec
import pytest
from jinja2.exceptions import SecurityError
from packaging.version import Version


def _load_references(tmp_path, manifest):
    path = tmp_path / "references.json"
    path.write_text(json.dumps(manifest))
    refs = json.loads(path.read_text())
    return fsspec.filesystem(
        "reference",
        fo=refs,
        simple_templates=not bool(refs.get("gen")),
        skip_instance_cache=True,
    )


def _generated(key):
    return {
        "version": 1,
        "refs": {},
        "gen": [{"dimensions": {}, "key": key, "url": "/unused"}],
    }


def test_fixed_fsspec_version():
    assert Version(version("fsspec")) >= Version("2026.6.0")


def test_static_references(tmp_path):
    fs = _load_references(tmp_path, {"version": 1, "refs": {"metadata": "plain text"}})
    assert fs.references["metadata"] == "plain text"


def test_generated_references(tmp_path):
    fs = _load_references(tmp_path, _generated("item_{{ 6 * 7 }}"))
    assert fs.references["item_42"] == ["/unused"]


def test_generated_reference_rejects_unsafe_attribute(tmp_path):
    with pytest.raises(SecurityError):
        _load_references(tmp_path, _generated("item_{{ ('sample').__class__.__name__ }}"))


def test_reference_url_rejects_unsafe_attribute(tmp_path):
    manifest = {
        "version": 1,
        "templates": {"base": "/unused"},
        "refs": {"item": ["path_{{ ('sample').__class__.__name__ }}"]},
        "gen": [{"dimensions": {}, "key": "static", "url": "/unused"}],
    }
    with pytest.raises(SecurityError):
        _load_references(tmp_path, manifest)


def test_template_definition_rejects_unsafe_attribute(tmp_path):
    manifest = {
        "version": 1,
        "templates": {"item": "path_{{ ('sample').__class__.__name__ }}"},
        "refs": {},
    }
    fs = _load_references(tmp_path, manifest)
    with pytest.raises(SecurityError):
        fs.templates["item"]()
