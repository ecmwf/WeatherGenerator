# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""FQN selection (globs), grouping (param | block | module_type) and layer sampling."""

from __future__ import annotations

from collections import defaultdict
from fnmatch import fnmatchcase

import torch

GROUP_BY_OPTIONS = ("param", "block", "module_type")

# wrapper prefixes that some PyTorch utilities insert into FQNs
_WRAPPER_PARTS = ("_checkpoint_wrapped_module", "_orig_mod", "_fsdp_wrapped_module")


def clean_fqn(fqn: str) -> str:
    parts = [p for p in fqn.split(".") if p not in _WRAPPER_PARTS]
    return ".".join(parts)


def block_prefix(fqn: str) -> str | None:
    """FQN up to and including the first integer component (`fe_blocks.3.mlp.0.w` ->
    `fe_blocks.3`); None if the FQN has no integer component."""
    parts = fqn.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[: i + 1])
    return None


def _parent(fqn: str) -> str:
    return fqn.rsplit(".", 1)[0] if "." in fqn else fqn


def _sample_indices(spec, num: int) -> set[int]:
    """Positions selected by `sample_layers` (ints, negative from the end, or "mid")."""
    selected = set()
    for s in spec:
        if isinstance(s, str):
            if s != "mid":
                raise ValueError(f"sample_layers entry '{s}' must be an int or 'mid'")
            selected.add(num // 2)
        elif -num <= int(s) < num:
            selected.add(int(s) % num)
    return selected


class Selection:
    """Which params / modules a monitor looks at and how they are grouped into metrics."""

    def __init__(self, include, exclude, group_by: str = "block", sample_layers=None):
        self.include = list(include) if include is not None else ["*"]
        self.exclude = list(exclude) if exclude is not None else []
        if group_by not in GROUP_BY_OPTIONS:
            raise ValueError(f"group_by must be one of {GROUP_BY_OPTIONS}, got '{group_by}'")
        self.group_by = group_by
        self.sample_layers = list(sample_layers) if sample_layers is not None else None

    def matches(self, fqn: str) -> bool:
        return any(fnmatchcase(fqn, p) for p in self.include) and not any(
            fnmatchcase(fqn, p) for p in self.exclude
        )

    def params(self, model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter, str]]:
        """Selected trainable params as (fqn, param, group)."""
        out = []
        for name, param in model.named_parameters():
            fqn = clean_fqn(name)
            if not param.requires_grad or not self.matches(fqn):
                continue
            if self.group_by == "param":
                group = fqn
            elif self.group_by == "block":
                group = block_prefix(fqn) or _parent(fqn)
            else:
                owner = model.get_submodule(_parent(name)) if "." in name else model
                group = f"{type(owner).__name__}.{fqn.rsplit('.', 1)[-1]}"
            out.append((fqn, param, group))
        return out

    def modules(
        self, model: torch.nn.Module, predicate=None, block_level_only: bool = False
    ) -> list[tuple[str, torch.nn.Module, str]]:
        """Selected modules as (fqn, module, group), after `sample_layers`.

        `block_level_only`: in `group_by: block` mode keep only modules that *are* a block
        (FQN ends at its first integer component), so that one group = one block output.
        """
        selected = []
        for name, module in model.named_modules():
            fqn = clean_fqn(name)
            if not fqn or not self.matches(fqn):
                continue
            if predicate is not None and not predicate(module):
                continue
            prefix = block_prefix(fqn)
            if self.group_by == "block" and block_level_only and prefix != fqn:
                continue
            selected.append((fqn, module))

        if self.sample_layers is not None:
            selected = self._sample(selected)

        out = []
        for fqn, module in selected:
            if self.group_by == "param":
                group = fqn
            elif self.group_by == "block":
                group = block_prefix(fqn) or fqn
            else:
                group = type(module).__name__
            out.append((fqn, module, group))
        return out

    def _sample(self, selected):
        """Keep the `sample_layers` positions of each block container (e.g. `fe_blocks`)."""
        indices = defaultdict(set)
        for fqn, _ in selected:
            prefix = block_prefix(fqn)
            if prefix is not None:
                container, idx = prefix.rsplit(".", 1) if "." in prefix else ("", prefix)
                indices[container].add(int(idx))
        keep = {}
        for container, idxs in indices.items():
            ordered = sorted(idxs)
            keep[container] = {
                ordered[i] for i in _sample_indices(self.sample_layers, len(ordered))
            }
        out = []
        for fqn, module in selected:
            prefix = block_prefix(fqn)
            if prefix is None:
                out.append((fqn, module))
                continue
            container, idx = prefix.rsplit(".", 1) if "." in prefix else ("", prefix)
            if int(idx) in keep[container]:
                out.append((fqn, module))
        return out
