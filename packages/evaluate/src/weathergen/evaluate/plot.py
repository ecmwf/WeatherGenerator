#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "weathergen-evaluate",
# ]
# [tool.uv.sources]
# weathergen-evaluate = { path = "../../../../../packages/evaluate" }
# ///


from weathergen.common import common_function


def plot():
    print(common_function())


if __name__ == "__main__":
    plot()
