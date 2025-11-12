#!/usr/bin/env -S uv run

# ruff: noqa: T201
"""
Checks that all pyproject.toml files are consistent for select sections
USAGE EXAMPLE: ./scripts/actions.sh toml-check from the root of the repo
"""

import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


def loop_keys(toml_dict, list_keys):
    for i in list_keys:
        toml_dict = toml_dict[i]
    return toml_dict


def check_toml_key(main_toml_dict, other_toml_dict, list_keys, name):
    try:
        main_value = loop_keys(dict(main_toml_dict), list_keys)
        other_value = loop_keys(dict(other_toml_dict), list_keys)
        assert main_value == other_value, (
            f"{list_keys} mismatch with main pyproject.toml and {name} pyproject.toml: ",
            f"{main_value} != {other_value}",
        )
    except Exception as e:
        assert (
            type(e) is not KeyError
        ), f"""KeyError: '{list_keys}' not found in {name} pyproject.toml, 
                please populate this field"""
        print(e)


def check_tomls(main_toml, *tomls):
    main_toml_dict = {}
    with open(main_toml, "rb") as toml_file:
        main_toml_dict = tomllib.load(toml_file)
    all_tomls = {}
    for toml in tomls:
        toml_dict = {}
        with open(toml, "rb") as toml_file:
            toml_dict = tomllib.load(toml_file)
        all_tomls[Path(toml)] = toml_dict
    for toml_path, toml_dict in all_tomls.items():
        # shorten name to package path
        name = toml_path.parent.name
        # check build system is the same
        check_toml_key(main_toml_dict, toml_dict, ["build-system"], name)
        # check python version is the same
        # check_toml_key(main_toml_dict, toml_dict, [], name)
        # check project.version/authors/urls are the same
        for key in ["version", "requires-python"]:
            check_toml_key(main_toml_dict["project"], toml_dict["project"], [key], name)
        # check tool.ruff is the same (disabled until issue 1081)
        # check_toml_key(main_toml_dict, toml_dict, ["tool", "ruff"], name)


if __name__ == "__main__":
    main_toml = _REPO_ROOT / "pyproject.toml"
    sub_packages = ["evaluate", "common", "metrics", "readers_extra"]
    tomls = [_REPO_ROOT / "packages" / package / "pyproject.toml" for package in sub_packages]
    check_tomls(main_toml, *tomls)
