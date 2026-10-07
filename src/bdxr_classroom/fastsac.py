"""Classroom runner around Holosoma's FastSAC networks and replay buffer.

Network, categorical critic projection, and replay implementations are imported
unchanged from the pinned Apache-2.0 Holosoma source. Update equations follow its
fast_sac_agent.py; this adapter supplies mjlab transitions and resumable logging.
"""

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor/holosoma/src/holosoma"))
from holosoma.agents.fast_sac.fast_sac import Actor, Critic
from holosoma.agents.fast_sac.fast_sac_utils import EmpiricalNormalization, SimpleReplayBuffer


class FastSAC:
    def __init__(self, actor_dim, critic_dim, action_dim, cfg):
        self.cfg = cfg
        self.dimensions = (actor_dim, critic_dim, action_dim)
        self.device = "cuda:0"
        ai = {"obs": {"start": 0, "end": actor_dim, "size": actor_dim}}
        ci = {"obs": {"start": 0, "end": critic_dim, "size": critic_dim}}
        self.actor = Actor(ai, ["obs"], action_dim, cfg["num_envs"],
                           cfg["actor_hidden"], 0.0, -5.0, device=self.device)
        self.critic = Critic(ci, ["obs"], action_dim, 101, -20, 20,
                             cfg["critic_hidden"], device=self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_norm = EmpiricalNormalization(actor_dim, device=self.device)
        self.critic_norm = EmpiricalNormalization(critic_dim, device=self.device)
        self.log_alpha = torch.tensor([math.log(0.001)], device=self.device, requires_grad=True)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=3e-4,
                                           betas=(0.9, 0.95), weight_decay=0.001, fused=True)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=3e-4,
                                            betas=(0.9, 0.95), weight_decay=0.001, fused=True)
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=3e-4,
                                           betas=(0.9, 0.95), fused=True)
        self.amp = cfg.get("amp", "off")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == "fp16")
        self.buffer = SimpleReplayBuffer(cfg["num_envs"], cfg["buffer_size"],
                                         actor_dim, action_dim, critic_dim,
                                         gamma=cfg["gamma"], device=self.device)
        self.update_count = 0
        self.update_critic_fn = self.update_critic
        self.update_actor_fn = self.update_actor
        if cfg.get("compile"):
            self.update_critic_fn = torch.compile(self.update_critic)
            self.update_actor_fn = torch.compile(self.update_actor)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        x = self.actor_norm(obs, update=False)
        if deterministic:
            return self.actor(x)[0]
        return self.actor.get_actions_and_log_probs(x)[0]

    def add(self, obs, actions, final, rewards, terminated, truncated):
        transition = TensorDict({
            "observations": obs["actor"], "critic_observations": obs["critic"],
            "actions": actions,
            "next": {
                "observations": final["actor"], "critic_observations": final["critic"],
                "rewards": rewards, "dones": (terminated | truncated).long(),
                # Simultaneous fall and timeout must never bootstrap a fall.
                "truncations": (truncated & ~terminated).long(),
            }}, batch_size=(self.cfg["num_envs"],), device=self.device)
        self.buffer.extend(transition)

    def update_critic(self, obs, actions, next_actor, next_critic, rewards, bootstrap):
        with torch.autocast("cuda", dtype=torch.float16 if self.amp == "fp16" else torch.bfloat16,
                            enabled=self.amp != "off"):
            with torch.no_grad():
                next_actions, next_logp = self.actor.get_actions_and_log_probs(next_actor)
                discount = torch.full_like(rewards, self.cfg["gamma"])
                distribution = self.target.projection(
                    next_critic, next_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_logp,
                    bootstrap, discount)
            outputs = self.critic(obs, actions)
            loss = -(distribution * F.log_softmax(outputs, -1)).sum(-1).mean(1).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.critic_opt)
        self.scaler.update()
        alpha_loss = (-self.log_alpha.exp() * next_logp.detach().float()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        return loss.detach(), alpha_loss.detach()

    def update_actor(self, obs, critic_obs):
        with torch.autocast("cuda", dtype=torch.float16 if self.amp == "fp16" else torch.bfloat16,
                            enabled=self.amp != "off"):
            actions, logp = self.actor.get_actions_and_log_probs(obs)
            values = self.critic.get_value(F.softmax(self.critic(critic_obs, actions), -1)).mean(0)
            loss = (self.log_alpha.exp().detach() * logp - values).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.actor_opt)
        self.scaler.update()
        return loss.detach(), -logp.detach().mean()

    def learn(self):
        metrics = {}
        for _ in range(self.cfg["updates"]):
            data = self.buffer.sample(max(1, self.cfg["batch_size"] // self.cfg["num_envs"]))
            actor = self.actor_norm(data["observations"])
            critic = self.critic_norm(data["critic_observations"])
            next_actor = self.actor_norm(data["next"]["observations"])
            next_critic = self.critic_norm(data["next"]["critic_observations"])
            bootstrap = (data["next"]["truncations"].bool() | ~data["next"]["dones"].bool()).float()
            qloss, aloss = self.update_critic_fn(
                critic, data["actions"], next_actor, next_critic,
                data["next"]["rewards"], bootstrap)
            metrics.update(critic_loss=qloss, alpha_loss=aloss)
            if self.update_count % 4 == 1:
                ploss, entropy = self.update_actor_fn(actor, critic)
                metrics.update(actor_loss=ploss, entropy=entropy)
            with torch.no_grad():
                torch._foreach_lerp_(list(self.target.parameters()), list(self.critic.parameters()), 0.125)
            self.update_count += 1
        metrics["alpha"] = self.log_alpha.exp().detach().mean()
        return metrics

    def save(self, path, step, include_replay=False):
        state = dict(config=self.cfg, dimensions=self.dimensions, step=step,
                     update_count=self.update_count,
                     log_alpha=self.log_alpha.detach(), rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(),
                     grad_scaler=self.scaler.state_dict(),
                     resume_note="Optimizer/normalizer continuation; simulation episodes restart.")
        for name in ["actor", "critic", "target", "actor_norm", "critic_norm",
                     "actor_opt", "critic_opt", "alpha_opt"]:
            state[name] = getattr(self, name).state_dict()
        if include_replay:
            state["replay"] = {k: v for k, v in vars(self.buffer).items()
                               if isinstance(v, torch.Tensor)}
            state["replay_ptr"] = self.buffer.ptr
        path = Path(path)
        temp = path.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(path)

    def load(self, path, replay=True):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("format") == "bdxr-inference-only-v1":
            raise ValueError("This is an inference-only policy; resume training from final.pt or resume.pt")
        if tuple(state["dimensions"]) != self.dimensions:
            raise ValueError("Checkpoint observation/action dimensions do not match this environment")
        for name in ["actor", "critic", "target", "actor_norm", "critic_norm",
                     "actor_opt", "critic_opt", "alpha_opt"]:
            getattr(self, name).load_state_dict(state[name])
        self.log_alpha.data.copy_(state["log_alpha"])
        if state.get("grad_scaler") and self.scaler.is_enabled():
            self.scaler.load_state_dict(state["grad_scaler"])
        self.update_count = state["update_count"]
        if replay and "replay" in state:
            for name, value in state["replay"].items():
                target = getattr(self.buffer, name)
                if target.shape != value.shape:
                    raise ValueError("Replay shapes differ; use --finetune to restart replay")
                target.copy_(value)
            self.buffer.ptr = state["replay_ptr"]
        torch.set_rng_state(state["rng"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        return state["step"]
