"""Single source of the branch research contract, also served by MCP."""

CONTRACT = """# Mahjong branch research

Use only mahjong_research MCP tools. No shell, other tools or subagents.
Read status, current representative and this branch's previous results and candidate history.
Fork any OWN historical candidate, modify its strategy.py package, and submit ONE batch of
1..configured_limit candidates (maximum four). End your reply immediately after submission.
The controller resumes this conversation after all prepared candidates finish equal-size
paired evaluation. You may return to an older parent; the comparison remains the incumbent.
Your branch's combination is binding: preserve its algorithm and risk style. You can change
implementation, architecture, parameters and training recipe inside those requirements.
COMBINATION must remain the exact combination ID. The object returned by build(model_dir)
must expose the following capability fields even if you replace the baseline or wrap it:
- heuristic: policy.style equals the combination style (balanced/adventurous/conservative),
  and policy.utilities(observation) is callable. PARAMETERS["style"] or base.style alone
  does NOT satisfy this interface; wrappers must expose their own style attribute.
- value/PPO: policy.algorithm equals lightgbm/mlp/ppo as configured; strategy.train(ctx)
  is callable; model/training.json records that algorithm and a positive updates count.
- search: policy.algorithm equals rollout or mcts as configured.
Use research_read_source("capabilities") for the complete validator. Preserve the actual
algorithm/behavior as well as these fields; a name or field alone does not establish it.

build(model_dir) returns reset(seed), act(observation), finish_round(result,seat).
Choose from observation.legal_actions. Optional train(ctx) produces model files in ctx.output.
ctx.initial contains the parent model; ctx.play(learner) runs a complete real settlement.
ctx.episodes bounds training games. ctx.checkpoint(metadata) atomically commits resumable
model state; ctx.completed_episodes and ctx.checkpoint_metadata describe recovery.
Do not complete a trainable model without actual training. Label actual chosen actions with
observed returns; label hypothetical search estimates separately. Preserve training.json with
algorithm and completed update count. Dependencies are exact name==version wheels.

Only legal observations and sanitized results enter candidates. No REAL hidden opponent hands,
active wall, environment random material, credentials or external platform feedback.
Sampling hypothetical hidden states from your observation using independent randomness IS
allowed; use the public simulation SDK. Never obtain the host's active true state.
Other branches' rankings are public, their source/models are inaccessible. All your own
historical versions remain available. Store your rationale in hypothesis and this conversation.

Each cycle freezes reference opponents. Every prepared candidate and incumbent complete the
same paired schedule; no early screening. Optimize average final rank utility 3,2,1,0.
The best evaluated version represents the branch. Cycles finish with shared competition and
protected branch elimination. Rankings update at cycle competitions, not after each batch.
You cannot change evaluator, schedule, resource slots, protection or research budget.
Default iteration execution limit is one hour, excluding queueing and framework evaluation.
Failures consume an attempt; successful partial batches are evaluated. Inspect actual errors.
Action timeouts default to 3 seconds for discards and 1 second for responses, with environment
fallback. Long simulations must respect the action window.
"""
