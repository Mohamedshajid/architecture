from __future__ import annotations

from .models import VerificationStatus


class VerificationGate:
    def decide(self, status: VerificationStatus) -> VerificationStatus:
        return status

    def should_reroute(self, status: VerificationStatus, attempt: int, max_attempts: int) -> bool:
        return status in (VerificationStatus.FAILED, VerificationStatus.UNCERTAIN) and attempt < max_attempts
