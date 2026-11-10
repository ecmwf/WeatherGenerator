"""Check the reference-mode switch needed by newer fsspec releases."""

import fsspec


def test_reference_modes():
    static = {"version": 1, "refs": {"metadata": "plain text"}}
    generated = {
        "version": 1,
        "refs": {},
        "gen": [{"dimensions": {}, "key": "item_{{ 6 * 7 }}", "url": "/unused"}],
    }
    for manifest, key, expected in (
        (static, "metadata", "plain text"),
        (generated, "item_42", ["/unused"]),
    ):
        fs = fsspec.filesystem(
            "reference",
            fo=manifest,
            simple_templates=not bool(manifest.get("gen")),
            skip_instance_cache=True,
        )
        assert fs.references[key] == expected
