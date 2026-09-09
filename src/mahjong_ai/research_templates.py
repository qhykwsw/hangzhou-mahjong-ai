"""Editable starting programs, not a restricted proposal language."""
RULE = '''from mahjong_ai.policy import RulePolicy

def build(model_dir):
    return RulePolicy()
'''

RL = '''from mahjong_ai.policy import RLPolicy, TrainingRecipe

RECIPE = TrainingRecipe()

def build(model_dir):
    return RLPolicy(RECIPE, model_dir / "model.npz")

def train(ctx):
    # Replace this entire function and model implementation to explore other RL algorithms.
    # ctx.play runs one real complete settlement and calls learner.finish_round once.
    # Save checkpoints under ctx.output; this directory survives interruption.
    initial = ctx.initial / "model.npz"
    checkpoint = ctx.output / "model.npz"
    load = checkpoint if ctx.completed_episodes else initial
    learner = RLPolicy(RECIPE, load if load.exists() else None, training=True)
    for episode in range(ctx.completed_episodes, ctx.episodes):
        result, seat = ctx.play(learner)
        learner.save(checkpoint)
        ctx.checkpoint({"episode": episode + 1})
    learner.save(ctx.output / "model.npz")
    return {"algorithm": RECIPE.algorithm, "episodes": ctx.episodes}
'''

CONTRACT = '''# Strategy research interface v1

Goal: maximize average terminal four-player rank utility (ranks 1/2/3/4 -> 3/2/1/0).
Use all legally observable information and variable settlement progress; no extra reward for
raw score at an unchanged final rank. Elo describes developer results, not promotion evidence.

Each candidate is a Python package containing strategy.py and optional helper modules.
build(model_dir) returns an object with reset(seed), act(observation), finish_round(result, seat).
act must return an element of observation.legal_actions. Imports, loops, new algorithms,
state and search are allowed. The worker only receives public observations and sanitized
terminal results. No actual unseen hands or future wall are available. Hidden-state
simulation must sample hypothetical worlds using public information.
observation.catch_play indicates a table-wide catch-play circle; god_discarder_seat is
its publicly visible exempt physical seat (-1 when inactive). catch_restricted indicates
whether this player must discard the drawn tile and cannot chi/peng/exposed or added kong.
The exempt player can claim and discard freely; use legal_actions as the action authority.

RL candidates additionally implement train(ctx). ctx.episodes is the allowed number of
real training settlements; ctx.play(learner) returns (result, seat), after invoking reset,
act and finish_round on learner. You can implement arbitrary RL optimizers, buffers,
networks, update schedules and checkpoint formats. Write model artifacts to ctx.output.
ctx.initial contains parent artifacts if available. ctx.output persists after interruption.
ctx.completed_episodes and ctx.checkpoint_metadata describe the last committed checkpoint.
After saving model/optimizer/buffer/RNG files under ctx.output, call ctx.checkpoint(metadata)
to commit them with real game progress. On resume load these files and continue from
ctx.completed_episodes. Without checkpoint calls, an interrupted train function restarts.
Training must use actual RL and the resulting model must be used by build.
Rule candidates must not load trained models. Both routes may use rules and search.

Frozen files are immutable. Create descendants to change code. Dependencies are declared
as exact package==version requirements; environment preparation and locking are host operations.
Use research_read_source to inspect SDK examples, observation types, rules and baseline code.

Workflow: status -> search history/get details -> choose parents -> create/read/write candidates
-> submit ONE batch of candidates and development experiments -> wait until all finish
-> inspect full batch and individual results -> record one unified population decision
-> optionally submit selected frozen candidates for independent acceptance -> next batch.
Do not evolve or change population while a batch is in progress. Failures are recorded.
All experiments remain searchable even after a candidate is archived. The environment
suggests parents; you may retain diverse candidates with a reason. Do not fabricate results.
Total active run time limits research. Formal acceptance slots are an optional cap;
acceptance_remaining=null means unlimited, not exhausted. Query remaining budget.
You cannot grant yourself additional time. Never use external competition data.
'''
