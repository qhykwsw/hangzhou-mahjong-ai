"""Row fields computed in the game worker.

The playout advantages (targets.py) need only whether the learner deals the
hand; no Critic is fitted, so no hidden-state features are computed.
"""


def dealer(state, seat):
    return {"is_dealer": float(state.dealer == seat)}
