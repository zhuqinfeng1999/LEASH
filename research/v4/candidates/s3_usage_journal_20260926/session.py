"""Persist SDK usage and messages while reusing the same S3 planner session.

This is logging only: no extra model calls, instruction changes or retries.
SDK thread usage is cumulative, so keep its latest snapshot rather than
adding snapshots from different rounds of the same thread.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

from research.v4.candidates.s3_v42_tabletop_20260926.model_session import S3CodexSession


def journal_recorder_type(base_type, event_path: Path, usage_path: Path, round_index: int):
    class JournalRecorder(base_type):
        def observe(self, event):
            rendered = super().observe(event)
            method = event.get('method', '') if isinstance(event, dict) else getattr(event, 'method', '')
            if rendered or method == 'thread/tokenUsage/updated':
                row = {'utc': datetime.now(timezone.utc).isoformat(),
                       'round': round_index, 'method': method,
                       'rendered': rendered[:8000], 'usage': dict(self.usage)}
                with event_path.open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            if method == 'thread/tokenUsage/updated':
                usage = dict(self.usage)
                usage['total_tokens'] = usage.get('total_input_tokens', 0) + usage.get('total_output_tokens', 0)
                record = {'source': 'actual SDK thread/tokenUsage/updated',
                          'cumulative_thread_usage': True, 'round': round_index,
                          'usage_tokens': usage}
                temporary = usage_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(record, indent=2) + '\n')
                temporary.replace(usage_path)
            return rendered
    return JournalRecorder


class JournaledS3CodexSession(S3CodexSession):
    async def run_round(self, turn_input, toolkit, boundary, *, max_turns, timeout_s):
        from rpent.planner import codex
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        kind = journal_recorder_type(codex._Recorder, output / 'planner-events.jsonl',
                                     output / 'planner-usage.json', turn_input.index)
        # The original persistent adapter imports this recorder in run_round.
        # Only its logger is replaced in this process; all decisions remain live.
        with patch.object(codex, '_Recorder', kind):
            return await super().run_round(turn_input, toolkit, boundary,
                                           max_turns=max_turns, timeout_s=timeout_s)
