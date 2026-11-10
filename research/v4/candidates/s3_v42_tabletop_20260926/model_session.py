"""Model-selectable S3 use of the existing persistent Codex session driver."""
from __future__ import annotations

from research.v4.candidates.multiround_bridge_20260924.driver import (
    CodexPersistentSession, PLANNER_TIMEOUT_S)


class S3CodexSession(CodexPersistentSession):
    def __init__(self,*,toolkit,output_dir,model_id='gpt-5.5',reasoning_effort='high'):
        super().__init__(toolkit=toolkit,output_dir=output_dir)
        self.model_id=model_id;self.reasoning_effort=reasoning_effort

    async def __aenter__(self):
        import openai_codex
        from research.v3.bridge.b0 import build_b0_planner
        from rpent.planner.utils.http_mcp_server import HttpMcpServer
        self._planner=build_b0_planner(output_dir=self.output_dir,toolkit=self.toolkit,
            timeout_s=PLANNER_TIMEOUT_S,model_id=self.model_id,
            reasoning_effort=self.reasoning_effort)
        self._server=HttpMcpServer(self.toolkit);url=self._server.start()
        try:
            self._codex=openai_codex.AsyncCodex(self._planner._build_config(url))
            self._thread=await self._codex.thread_start(ephemeral=True,**self._planner._turn_options)
            self.thread_starts=1
        except BaseException:
            self._server.stop();raise
        return self
