from __future__ import annotations

from grosscode.core import DecoderConfig, SideContext
from grosscode.decoders._base import _BaseSideDecoder


class FullBPDecoder(_BaseSideDecoder):
    def __init__(self, context: SideContext, config: DecoderConfig = DecoderConfig()) -> None:
        super().__init__(context=context, config=config, algorithm="bp")
