"""Train MADDPG on simple_tag_v3 and record rollout GIFs at chosen episodes.

Reuses the Actor/Critic/MADDPGAgent/ReplayBuffer classes and the update rule from
MADDPG.py, and adds a geometric (recency-biased) replay sampler, reward logging and
deterministic rollout snapshots so the learning progress can be seen as an animation.

Example:
    python visualize_training.py --out results/geom --episodes 4000 --sampler geometric
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pettingzoo.mpe import simple_tag_v3

from MADDPG import MADDPGAgent, ReplayBuffer


class GeometricReplayBuffer(ReplayBuffer):
    """Replay buffer that draws reverse indices k ~ Geom(p), so index N-k favours recent transitions."""

    def __init__(self, capacity, p):
        super().__init__(capacity)
        self.p = p

    def sample(self, batch_size):
        n = len(self.buffer)
        k = np.random.geometric(self.p, size=batch_size)  # k = 1 is the newest transition
        idx = n - np.minimum(k, n)
        return [self.buffer[i] for i in idx]


def rollout(agents, ep_len, seed):
    """Run one deterministic episode and return rendered frames plus per-agent return."""
    env = simple_tag_v3.parallel_env(render_mode="rgb_array", max_cycles=ep_len)
    obs, _ = env.reset(seed=seed)
    frames, total = [], {a: 0.0 for a in env.agents}
    while env.agents:
        acts = {a: agents[a].get_action(obs[a], deterministic=True) for a in env.agents}
        obs, rew, _, _, _ = env.step(acts)
        for a, r in rew.items():
            total[a] += r
        frames.append(Image.fromarray(env.render()))
    env.close()
    return frames, total


def update(agents, all_agents, env, buffer, batch, gamma, tau, device):
    """One MADDPG update for every agent (same rule as MADDPG.train)."""
    for adv in all_agents:
        samples = buffer.sample(batch)
        s_states = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in samples]).to(device)
        s_next = torch.stack([torch.tensor(s[3], dtype=torch.float32) for s in samples]).to(device)
        s_rew = torch.tensor([s[2][adv] for s in samples], dtype=torch.float32).to(device)
        next_acts, joint_acts, actor_acts = [], [], []
        for ag in all_agents:
            obs_b = torch.stack([torch.tensor(s[4][ag], dtype=torch.float32) for s in samples]).to(device)
            with torch.no_grad():
                nxt = torch.argmax(agents[ag].target_actor(obs_b), dim=-1)
            next_acts.append(F.one_hot(nxt, env.action_space(ag).n).float())
            a = torch.tensor([s[1][ag] for s in samples], dtype=torch.int64)
            joint_acts.append(F.one_hot(a, env.action_space(ag).n).float().to(device))
            logits = agents[ag].actor(obs_b)
            if ag == adv:
                actor_acts.append(logits)
                own_logits = logits
            else:
                actor_acts.append(logits.detach())
        with torch.no_grad():
            target_q = s_rew + gamma * agents[adv].target_critic(s_next, torch.cat(next_acts, -1)).squeeze(1)
        current_q = agents[adv].critic(s_states, torch.cat(joint_acts, -1)).squeeze(1)
        agents[adv].update_critic(F.mse_loss(current_q, target_q))
        actor_loss = -agents[adv].critic(s_states, torch.cat(actor_acts, -1)).mean()
        agents[adv].update_actor(actor_loss + 1e-3 * (own_logits ** 2).mean())
    for adv in all_agents:
        agents[adv].update_targets(tau=tau)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory for GIFs, actor weights and rewards.json")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--ep-len", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--capacity", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=1.0, help="initial exploration noise on logits")
    ap.add_argument("--noise-min", type=float, default=0.1)
    ap.add_argument("--update-every", type=int, default=2, help="env steps between gradient updates")
    ap.add_argument("--warmup", type=int, default=100, help="random-action episodes before learning starts")
    ap.add_argument("--sampler", choices=["uniform", "geometric"], default="uniform")
    ap.add_argument("--geom-p", type=float, default=1e-4, help="geometric parameter p (larger = stronger recency bias)")
    ap.add_argument("--snapshots", default="0,500,1000,2000,3000,4000", help="episodes at which to record a rollout GIF")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cpu"

    env = simple_tag_v3.parallel_env(render_mode=None, max_cycles=args.ep_len)
    obs, _ = env.reset(seed=args.seed)
    all_agents = list(env.agents)
    joint = sum(env.action_space(a).n for a in all_agents)
    agents = {a: MADDPGAgent(len(obs[a]), env.action_space(a).n, len(env.state()), joint, args.lr, device)
              for a in all_agents}
    if args.sampler == "geometric":
        buffer = GeometricReplayBuffer(args.capacity, args.geom_p)
    else:
        buffer = ReplayBuffer(args.capacity)
    snaps = {int(s) for s in args.snapshots.split(",")}
    log = {a: [] for a in all_agents}
    t0, step = time.time(), 0

    def snapshot(ep):
        frames, total = rollout(agents, args.ep_len, seed=123)
        frames[0].save(os.path.join(args.out, f"rollout_ep{ep:05d}.gif"), save_all=True,
                       append_images=frames[1:], duration=100, loop=0)
        torch.save({a: ag.actor.state_dict() for a, ag in agents.items()},
                   os.path.join(args.out, f"actors_ep{ep:05d}.pt"))
        print(f"[snapshot] episode {ep} eval return {total}", flush=True)

    for ep in range(args.episodes + 1):
        if ep in snaps:
            snapshot(ep)
        if ep == args.episodes:
            break
        obs, _ = env.reset()
        ep_rew = {a: 0.0 for a in all_agents}
        frac = min(1.0, ep / max(1, args.episodes * 0.6))
        noise = args.noise + (args.noise_min - args.noise) * frac
        while env.agents:
            state = env.state()
            if ep < args.warmup:
                acts = {a: env.action_space(a).sample() for a in all_agents}
            else:
                acts = {a: agents[a].get_action(obs[a], noise_param=noise) for a in all_agents}
            next_obs, rew, _, _, _ = env.step(acts)
            buffer.push(state, acts, rew, env.state(), obs)
            for a, r in rew.items():
                ep_rew[a] += r
            obs = next_obs
            step += 1
            if ep >= args.warmup and len(buffer.buffer) >= args.batch and step % args.update_every == 0:
                update(agents, all_agents, env, buffer, args.batch, args.gamma, args.tau, device)
        for a in all_agents:
            log[a].append(ep_rew[a])
        if (ep + 1) % 50 == 0:
            avg = {a: float(np.mean(log[a][-50:])) for a in all_agents}
            print(f"ep {ep + 1:5d} noise {noise:.2f} {time.time() - t0:6.0f}s avg50 "
                  + " ".join(f"{a}={v:6.2f}" for a, v in avg.items()), flush=True)
            with open(os.path.join(args.out, "rewards.json"), "w") as f:
                json.dump(log, f)


if __name__ == "__main__":
    main()
