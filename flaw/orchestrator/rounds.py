"""Round state machine: phases, elastic close rules, and voiding (spec 5.5).

The orchestrator drives one :class:`RoundManager` per run.  Each federated
round moves ``IDLE`` -> ``OPEN`` -> ``IDLE`` (with a final ``FINISHED`` at
the end of the run): :meth:`RoundManager.open` freezes the snapshot of nodes
expected to report this round, :meth:`RoundManager.submit` accepts one result
per node, and :meth:`RoundManager.close_reason` decides — from the liveness
view passed in — whether the round can close early (``"all_reported"``) or
must wait for the deadline.  A round that collects fewer than
``min_participants`` results is *void* and is not aggregated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence, Set
from dataclasses import dataclass
from enum import StrEnum

from flaw.protocol import BlobRef


class RoundPhase(StrEnum):
    """Stored lifecycle phase of a round (``closing`` is transient in the server)."""

    IDLE = "idle"
    OPEN = "open"
    FINISHED = "finished"


class SubmitOutcome(StrEnum):
    """Why a submitted result was accepted or rejected."""

    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE = "stale"
    UNKNOWN_ROUND = "unknown_round"


@dataclass
class ParticipantResult:
    """One node's contribution to a round."""

    node_id: str
    num_samples: int
    metrics: dict[str, float]
    duration_s: float
    device: str
    update_blob: BlobRef


@dataclass
class RoundResult:
    """Outcome of a closed round; a void round carries no ``closed_reason``."""

    round: int
    participants: list[ParticipantResult]
    closed_reason: str | None
    void: bool
    void_reason: str | None


class RoundManager:
    """Phase FSM for one run's rounds: open, collect, close, repeat."""

    def __init__(
        self,
        min_participants: int,
        round_deadline_s: float,
        clock: Callable[[], float],
    ) -> None:
        self.min_participants = min_participants
        self.round_deadline_s = round_deadline_s
        self._clock = clock
        self._phase = RoundPhase.IDLE
        self._round = 0
        self._expected: list[str] = []
        self._opened_at = 0.0
        self._participants: dict[str, ParticipantResult] = {}

    @property
    def phase(self) -> RoundPhase:
        """Current phase (starts :attr:`RoundPhase.IDLE`)."""
        return self._phase

    @property
    def current_round(self) -> int:
        """Round number currently open (0 before the first :meth:`open`)."""
        return self._round

    def open(self, round_no: int, expected: Sequence[str]) -> None:
        """Open *round_no*, freezing the expected-participant snapshot.

        Re-callable after :meth:`close` with the same round number (the
        server re-opens a voided round); each call resets the snapshot, the
        open time, and the collected results.
        """
        self._round = round_no
        self._expected = list(expected)
        self._opened_at = self._clock()
        self._participants = {}
        self._phase = RoundPhase.OPEN

    def submit(self, node_id: str, round_no: int, participant: ParticipantResult) -> SubmitOutcome:
        """Record *participant* for *round_no*, classifying any rejection.

        A submission from a node outside the expected snapshot is still
        accepted — ``expected`` only governs close timing.
        """
        if self._phase is not RoundPhase.OPEN:
            return SubmitOutcome.STALE
        if round_no > self._round:
            return SubmitOutcome.UNKNOWN_ROUND
        if round_no < self._round:
            return SubmitOutcome.STALE
        if node_id in self._participants:
            return SubmitOutcome.DUPLICATE
        self._participants[node_id] = participant
        return SubmitOutcome.ACCEPTED

    def close_reason(self, alive_expected: Set[str]) -> str | None:
        """Decide whether the open round can close now, or ``None`` to keep waiting.

        ``"all_reported"`` once every alive expected node has reported (an
        empty *alive_expected* never closes this way); otherwise
        ``"deadline"`` once the round has run for ``round_deadline_s``.
        """
        if alive_expected and all(node in self._participants for node in alive_expected):
            return "all_reported"
        if self._clock() - self._opened_at >= self.round_deadline_s:
            return "deadline"
        return None

    def close(self, reason: str) -> RoundResult:
        """Close the open round, build its result, and return to ``IDLE``.

        The round is void (excluded from aggregation) when fewer than
        ``min_participants`` nodes reported: the result then carries
        ``void_reason="insufficient_participants"`` and no ``closed_reason``.
        """
        participants = sorted(self._participants.values(), key=lambda p: p.node_id)
        void = len(participants) < self.min_participants
        result = RoundResult(
            round=self._round,
            participants=participants,
            closed_reason=None if void else reason,
            void=void,
            void_reason="insufficient_participants" if void else None,
        )
        self._phase = RoundPhase.IDLE
        return result

    def finish(self) -> None:
        """Mark the run finished; further submissions are stale."""
        self._phase = RoundPhase.FINISHED

    def expected(self) -> list[str]:
        """Snapshot of the nodes expected this round, in insertion order."""
        return list(self._expected)

    def submitted_nodes(self) -> list[str]:
        """Ids of the nodes that reported this round, sorted."""
        return sorted(self._participants)
