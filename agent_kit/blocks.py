"""Putting prompt blocks into a module's prompt without the config having to name them.

Pieces built around a model add to what it is shown - the tools it may ask for, the
memory of its earlier attempts - and each does it through a block that is empty unless
there is something to say. A config that has never heard of them is left as it was; the
piece that needs them installs them in the module's builder, before `output_format`.
"""
from __future__ import annotations

from typing import Sequence

from llm_kit.prompt_builder import BlockSpec, PromptBuilder


def with_blocks(config, names: Sequence[str]):
    """`config` with the blocks it does not have yet added before `output_format` (at the end without one)."""
    present = [BlockSpec.parse(spec).name for spec in config.blocks]
    missing = [name for name in names if name not in present]
    if not missing:
        return config
    blocks = list(config.blocks)
    at = present.index("output_format") if "output_format" in present else len(blocks)
    return config.model_copy(update={"blocks": blocks[:at] + missing + blocks[at:]})


def install_blocks(module, names: Sequence[str]) -> None:
    """Rebuild `module.builder` with `names` among its blocks, keeping the resolvers and filters it was built with."""
    old = module.builder
    wider = with_blocks(old.config, names)
    if wider is not old.config:
        module.builder = PromptBuilder(wider, old.tokenizer, resolver_registry=old.resolver_registry,
                                       filter_registry=old.filter_registry)
