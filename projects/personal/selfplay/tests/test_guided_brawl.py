from collections import Counter

import pytest

from guided.brawl import schedule, summarize, validate_records
from guided.resources import choose_resources


def test_ten_entrants_balance_all_210_lineups_and_every_seat():
    entrants = list("ABCDEFGHIJ")
    jobs = schedule(entrants, 256, 17)
    assert len(jobs) == 10240
    seats = Counter((e, s) for j in jobs for s, e in enumerate(j["seats"]))
    assert all(seats[e, s] == 1024 for e in entrants for s in range(4))
    lineups = Counter(tuple(sorted(j["seats"])) for j in jobs)
    assert len(lineups) == 210 and set(lineups.values()) == {48, 52}
    records = [j | {"scores": [3, 1, -1, -3]} for j in jobs]
    validate_records(records, jobs, entrants)
    assert all(
        r["appearances"] == 4096 and r["mean"] == 0
        for r in summarize(records, entrants)["results"].values()
    )


def test_five_entrants_play_every_lineup_each_block():
    entrants = list("ABCDE")
    jobs = schedule(entrants, 128, 17)
    assert len(jobs) == 128 * 20
    seats = Counter((e, s) for j in jobs for s, e in enumerate(j["seats"]))
    assert all(seats[e, s] == 128 * 4 for e in entrants for s in range(4))
    records = [j | {"scores": [3, 1, -1, -3]} for j in jobs]
    validate_records(records, jobs, entrants)
    results = summarize(records, entrants)["results"]
    assert all(r["appearances"] == 2048 for r in results.values())


def test_eleven_entrants_balance_all_330_lineups_and_every_seat():
    entrants = list("ABCDEFGHIJK")
    jobs = schedule(entrants, 60, 17)
    assert len(jobs) == 60 * 44
    seats = Counter((e, s) for j in jobs for s, e in enumerate(j["seats"]))
    assert all(seats[e, s] == 60 * 4 for e in entrants for s in range(4))
    lineups = Counter(tuple(sorted(j["seats"])) for j in jobs)
    assert len(lineups) == 330 and set(lineups.values()) == {8}
    records = [j | {"scores": [3, 1, -1, -3]} for j in jobs]
    validate_records(records, jobs, entrants)


def test_resources_follow_cpu_and_memory_without_host_names():
    fast = choose_resources(20, 20 * 2**30)
    small = choose_resources(8, 5 * 2**30)
    assert fast["workers"] == 16 and fast["tables"] == 64
    assert small["workers"] == 6 and small["tables"] == 32
    assert choose_resources(1, 2**30)["workers"] == 1


def test_balanced_eight_policy_tournament():
    entrants = list("ABCDEFGH")
    jobs = schedule(entrants, 1024, 17)
    assert len(jobs) == 8192
    assert jobs == schedule(entrants, 1024, 17)
    seats = Counter((e, s) for j in jobs for s, e in enumerate(j["seats"]))
    assert all(seats[e, s] == 1024 for e in entrants for s in range(4))
    lineups = Counter(tuple(sorted(j["seats"])) for j in jobs)
    assert len(lineups) == 70
    assert set(lineups.values()) == {116, 120}
    assert {j["seed"] for j in jobs}.isdisjoint(
        j["seed"] for j in schedule(entrants, 1024, 18)
    )


def test_all_seats_count_and_incomplete_receipts_rejected():
    entrants = list("ABCDEFGH")
    jobs = schedule(entrants, 4, 17)
    records = [j | {"scores": [3, 1, -1, -3]} for j in jobs]
    validate_records(records, jobs, entrants)
    summary = summarize(records, entrants)
    assert all(
        r["appearances"] == 16 and r["mean"] == 0 for r in summary["results"].values()
    )
    assert summary["paired_differences"]["A"]["B"]["mean"] == 0
    with pytest.raises(ValueError, match="schedule"):
        validate_records(records[:-1], jobs, entrants)
    with pytest.raises(ValueError, match="schedule"):
        validate_records(records + [records[0]], jobs, entrants)
    records[0]["scores"] = [0, 0, 0, 1]
    with pytest.raises(ValueError, match="zero-sum"):
        validate_records(records, jobs, entrants)
