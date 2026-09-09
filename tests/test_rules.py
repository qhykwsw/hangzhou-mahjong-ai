import json
from pathlib import Path

import pytest

from mahjong_ai.rules import can_hu, calculate_fan, is_baotou, payments, tile_id, counts


def hand(text): return [tile_id(t) for t in text.split()]


def test_official_four_white_oracle():
    oracle = json.loads(Path('docs/sources/fan-calc-four-whites.json').read_text())
    request = oracle['request']
    h = [tile_id(t) for t in request['hand']]
    draw = tile_id(request['draw'])
    fan = calculate_fan(h + [draw], draw)
    assert fan and fan.fan == oracle['response']['fan'] == 8
    assert fan.baotou
    assert set(fan.detail) == set(oracle['response']['detail'])
    assert payments(0, 0, fan.fan) == (192, -64, -64, -64)
    assert payments(1, 0, fan.fan) == (-64, 80, -8, -8)


def test_standard_open_and_seven_pairs_are_distinct():
    normal = hand('1w 1w 1w 2w 3w 4w 5w 6w 7w 8w 9w 东 东 东')
    assert can_hu(normal)
    assert not can_hu(normal[:-1])
    assert can_hu(hand('2w 3w 4w 5w 6w 7w 8w 9w 白 东 东'), 1)
    seven = hand('1w 1w 2w 2w 3w 3w 4w 4w 5b 5b 东 东 白 白')
    assert can_hu(seven)
    assert not can_hu(seven, 1)
    assert not can_hu(hand('1w 2w 4w 5w 7w 8w 1b 2b 4b 5b 东 南 西 北'))


def test_luxury_whites_not_double_counted_when_used_as_jokers():
    h = hand('1w 1w 2w 2w 3w 3w 4w 5w 6w 白 白 白 白 9t')
    assert calculate_fan(h, tile_id('9t')).fan == 8  # seven x whites x baotou
    natural = hand('1w 1w 2w 2w 3w 3w 4w 4w 5b 5b 白 白 白 白')
    fan = calculate_fan(natural, tile_id('5b'))
    assert fan and '豪华七对' in fan.detail


def test_self_draw_gate_and_you_cai_configuration():
    h = hand('1w 1w 1w 2w 3w 4w 5w 6w 7w 8w 9w 东 东 白')
    assert calculate_fan(h, None) is None
    # Find a legal winning hand with a joker which is not an all-tile wait.
    result = calculate_fan(h, tile_id('1w'))
    assert result and not result.baotou
    assert calculate_fan(h, tile_id('1w'), you_cai_bi_kao=True) is None
    assert calculate_fan(h, tile_id('1w'), you_cai_bi_kao=True, kong_draw=True, chain_count=1)


@pytest.mark.parametrize('dealer', range(4))
@pytest.mark.parametrize('winner', range(4))
def test_score_is_zero_sum(dealer, winner):
    delta = payments(winner, dealer, 16, 3)
    assert sum(delta) == 0
    assert delta[winner] == (24 if winner == dealer else 10) * 16 * 3
