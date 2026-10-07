import torch

from flaw.orchestrator.rounds import ParticipantResult, RoundManager, RoundPhase, SubmitOutcome
from flaw.protocol import BlobRef, serialize_state_dict


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def participant(node_id: str, n: int = 10) -> ParticipantResult:
    sd = {"w": torch.ones(1)}
    data = serialize_state_dict(sd)
    return ParticipantResult(
        node_id=node_id,
        num_samples=n,
        metrics={"loss": 0.5, "val_accuracy": 0.5},
        duration_s=1.0,
        device="cpu",
        update_blob=BlobRef(blob_id=f"r0001/node-{node_id}", sha256="0" * 64, size=len(data)),
    )


def make(min_participants: int = 1, deadline: float = 10.0) -> tuple[RoundManager, FakeClock]:
    clock = FakeClock()
    return RoundManager(
        min_participants=min_participants, round_deadline_s=deadline, clock=clock
    ), clock


def test_open_sets_state() -> None:
    mgr, _ = make()
    assert mgr.phase is RoundPhase.IDLE and mgr.current_round == 0
    mgr.open(1, ["a", "b"])
    assert mgr.phase is RoundPhase.OPEN and mgr.current_round == 1
    assert mgr.expected() == ["a", "b"]


def test_submit_outcomes() -> None:
    mgr, _ = make()
    mgr.open(2, ["a"])
    assert mgr.submit("a", 1, participant("a")) is SubmitOutcome.STALE
    assert mgr.submit("a", 3, participant("a")) is SubmitOutcome.UNKNOWN_ROUND
    assert mgr.submit("a", 2, participant("a")) is SubmitOutcome.ACCEPTED
    assert mgr.submit("a", 2, participant("a")) is SubmitOutcome.DUPLICATE
    assert mgr.submit("b", 2, participant("b")) is SubmitOutcome.ACCEPTED  # not expected: still ok
    assert mgr.submitted_nodes() == ["a", "b"]


def test_submit_after_finish_is_stale() -> None:
    mgr, _ = make()
    mgr.open(1, ["a"])
    mgr.finish()
    assert mgr.submit("a", 1, participant("a")) is SubmitOutcome.STALE


def test_close_reason_all_reported() -> None:
    mgr, _ = make()
    mgr.open(1, ["a", "b"])
    mgr.submit("a", 1, participant("a"))
    assert mgr.close_reason({"a", "b"}) is None
    mgr.submit("b", 1, participant("b"))
    assert mgr.close_reason({"a", "b"}) == "all_reported"


def test_dead_node_does_not_block() -> None:
    mgr, _ = make()
    mgr.open(1, ["a", "b"])
    mgr.submit("a", 1, participant("a"))
    assert mgr.close_reason({"a"}) == "all_reported"  # b is dead → not awaited


def test_deadline_close() -> None:
    mgr, clock = make(deadline=10.0)
    mgr.open(1, ["a", "b"])
    assert mgr.close_reason({"a", "b"}) is None
    clock.now = 10.0
    assert mgr.close_reason({"a", "b"}) == "deadline"


def test_empty_alive_set_waits_for_deadline() -> None:
    mgr, clock = make(deadline=10.0)
    mgr.open(1, ["a"])
    assert mgr.close_reason(set()) is None
    clock.now = 10.0
    assert mgr.close_reason(set()) == "deadline"


def test_close_result_and_phase_reset() -> None:
    mgr, _ = make(min_participants=2)
    mgr.open(1, ["a", "b"])
    mgr.submit("b", 1, participant("b"))
    mgr.submit("a", 1, participant("a"))
    result = mgr.close("all_reported")
    assert [p.node_id for p in result.participants] == ["a", "b"]  # sorted
    assert not result.void and result.closed_reason == "all_reported"
    assert result.round == 1
    assert mgr.phase is RoundPhase.IDLE


def test_void_when_below_min_participants() -> None:
    mgr, _ = make(min_participants=2)
    mgr.open(1, ["a", "b"])
    mgr.submit("a", 1, participant("a"))
    result = mgr.close("deadline")
    assert result.void and result.void_reason == "insufficient_participants"
    assert result.closed_reason is None
