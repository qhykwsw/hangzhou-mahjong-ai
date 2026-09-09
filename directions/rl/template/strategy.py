from mahjong_ai.policy import RLPolicy, TrainingRecipe

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
