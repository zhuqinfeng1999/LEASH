"""Existing independent live reviewer, with neutral T1 noun schema only."""
from unittest.mock import patch
from research.v4.candidates.p4_natural_method_20260926 import review_worker as existing
from .public_boundary import PUBLIC_NOUNS
class T1ReviewerWorker(existing.P4ReviewerWorker):
    def review(self,gate,packet):
        with patch.object(existing,'PROMPTS',PUBLIC_NOUNS):return super().review(gate,packet)
