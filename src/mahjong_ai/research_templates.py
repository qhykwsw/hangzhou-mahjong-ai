"""Small research interface shared by every configured direction."""
CONTRACT = '''# Mahjong strategy research

Optimize average complete four-player settlement rank utility: 3, 2, 1, 0.
Use ONLY mahjong_research MCP tools. No native shell, other tools or subagents.
The direction brief is an initial research suggestion, not a restricted algorithm type.
Read status and your previous complete batch evidence; edit your own candidate package;
submit one batch of one to four candidates, then END this turn with a short explanation.
The controller will resume this same conversation when that batch has finished.

Candidates implement strategy.py: build(model_dir) returns reset(seed), act(observation),
finish_round(result,seat). Choose an action in observation.legal_actions. Optional train(ctx)
may generate models or other artifacts; ctx.play(learner) executes a real settlement.
ctx.episodes bounds training games. Save artifacts in ctx.output; parent model files are
in ctx.initial. ctx.checkpoint(metadata) saves a resumable training checkpoint. No family
name determines what algorithms or models are allowed. Dependencies are exact version wheels.

reset seeds are private-independent strategy randomness, never wall seeds. Only legal
observations and sanitized results enter training. No hidden opponent hands, future wall,
platform credentials, other directions' source or models. You may inspect common rankings,
but may only read, fork and edit candidates owned by this direction.

The arena chooses opponents and sample counts. Each period has a frozen diverse roster,
balanced appearances and four-seat rotations. Results are available after the whole period.
Lowest average rank utility entrants are periodically archived; history remains readable.
There is one period-relative leaderboard, no separate acceptance or certification.
Timeouts use official fallback and affect actual results, without extra ranking penalties.
The configured action windows default to 3 seconds for discards and 1 second for responses.

You cannot change the evaluator, pool membership, match conditions, worker limit or budget.
Do not create workspace experiments or structured knowledge records. Use candidate hypotheses
and your concise final explanation to preserve the reason for a change. Fork a new candidate
to change a frozen package. Initial draft is named initial within your direction.
'''
