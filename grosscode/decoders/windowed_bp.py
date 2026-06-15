from __future__ import annotations

from grosscode.core import DecoderConfig, SideContext, WindowConfig
from grosscode.decoders._base import _WindowedSideDecoder


class WindowedBPDecoder(_WindowedSideDecoder):
    def __init__(
        self,
        context: SideContext,
        config: DecoderConfig = DecoderConfig(),
        window: WindowConfig = WindowConfig(window_size=512, overlap_size=128),
    ) -> None:
        super().__init__(context=context, config=config, algorithm="bp", window=window)
