"""The documentation must not label an outdated checkout as synchronized."""

import pytest

from scripts.docs import gen_facts


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ("0\t0", "同步"),
        ("36\t0", "本地落后 36"),
        ("0\t2", "本地领先 2"),
        ("36\t2", "本地落后 36，领先 2（已分叉）"),
    ],
)
def test_git_facts_distinguish_behind_ahead_and_diverged(monkeypatch, counts, expected):
    def run_git(args):
        if args[:2] == ["rev-list", "--left-right"]:
            return counts
        return "test-ref"

    monkeypatch.setattr(gen_facts, "run_git", run_git)
    assert gen_facts.collect_git_facts()["gaps"] == {
        "origin": expected,
        "gitee": expected,
    }
