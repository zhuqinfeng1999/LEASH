"""Module-level spawn classes; unchanged independent SDK reviewer transport."""
from research.v4.candidates.leash_reviewer_transport_20260924.transport import SDKReviewerWorker
from .factory import components
class OwnS1Reviewer(SDKReviewerWorker):
    INSTANCE=None
    def review(self,gate,packet):
        return components(self.INSTANCE).worker.S1SDKReviewerWorker.review(self,gate,packet)
class S101Reviewer(OwnS1Reviewer):INSTANCE='S1-01'
class S102Reviewer(OwnS1Reviewer):INSTANCE='S1-02'
class S103Reviewer(OwnS1Reviewer):INSTANCE='S1-03'
class S104Reviewer(OwnS1Reviewer):INSTANCE='S1-04'
class S105Reviewer(OwnS1Reviewer):INSTANCE='S1-05'
WORKERS={c.INSTANCE:c for c in (S101Reviewer,S102Reviewer,S103Reviewer,S104Reviewer,S105Reviewer)}
