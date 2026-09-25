"""Replay the complete fixed Critic update, including gradient clipping and Adam."""

import copy

import torch
from torch.nn import functional as F


class CriticSteps:
    def __init__(self, model, learning_rate):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, eps=1e-5, capturable=True
        )
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        self.cache, self.pool = {}, None

    def _step(self, batch, goal):
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            pred = self.model(batch)
        loss = ((pred.float() - goal) ** 2).mean()
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

    def __call__(self, batch, goal):
        width = -(-batch["events"].shape[1] // 32) * 32
        batch = dict(batch, events=F.pad(batch["events"], (0, 0, 0, width - batch["events"].shape[1])))
        key = (len(goal), width)
        if key not in self.cache:
            static = {name: value.clone() for name, value in batch.items()}
            target = goal.clone()
            # Warm-up executes real updates. Restore parameters and all Adam
            # moments before capture, preserving the persistent tensor addresses.
            saved_model = copy.deepcopy(self.model.state_dict())
            saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._step(static, target)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.model.load_state_dict(saved_model)
            if saved_optimizer["state"]:
                for index, parameter in enumerate(self.model.parameters()):
                    for name, value in saved_optimizer["state"][index].items():
                        self.optimizer.state[parameter][name].copy_(value)
            else:
                for state in self.optimizer.state.values():
                    for value in state.values():
                        value.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._step(static, target)
            self.pool = graph.pool()
            self.cache[key] = static, target, graph
        static, target, graph = self.cache[key]
        for name, value in batch.items():
            static[name].copy_(value)
        target.copy_(goal)
        graph.replay()
