import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import copy


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    x = torch.clamp(x, min=-20.0, max=20.0)
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class RSSMCore(nn.Module):
    """
    Recurrent State-Space Model (RSSM) faithful to MuDreamer/DreamerV2.

    State s_t = {h_t, z_t} where:
      h_t : deterministic recurrent hidden state (GRU output), shape (B, deter_dim)
      z_t : stochastic categorical latent (straight-through gradients),
            shape (B, stoch_dim * stoch_classes)

    Two forward paths:
      Posterior  q(z_t | h_t, x_t) -- uses LeWM obs embedding (real env / WM training)
      Prior      p(z_t | h_t)       -- no observation, used during imagination rollouts

    KL(posterior || prior) shapes the latent so the prior can predict it,
    enabling coherent imagined trajectories.
    BatchNorm inside the posterior network prevents representation collapse
    without a pixel reconstruction loss (MuDreamer paper Section 4.1).
    """

    def __init__(
        self,
        obs_embed_dim=512,
        action_dim=4,
        deter_dim=256,
        stoch_dim=16,
        stoch_classes=16,
        hidden_dim=256,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_flat_dim = stoch_dim * stoch_classes
        self.state_dim = deter_dim + self.stoch_flat_dim

        # 1. Sequential network — deterministic path
        self.gru = nn.GRUCell(self.stoch_flat_dim + action_dim, deter_dim)

        # 2. Representation network — posterior q(z_t | h_t, x_t)
        # BatchNorm prevents collapse without pixel reconstruction loss
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + obs_embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

        # 3. Dynamics predictor — prior p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )

        # 4. Prediction heads — read from full state s_t = cat(h_t, z_t)
        self.reward_head = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_flat_dim, device=device)
        return h, z

    def _sample_straight_through(self, logits):
        B = logits.shape[0]
        logits_2d = logits.view(B, self.stoch_dim, self.stoch_classes)
        soft = F.softmax(logits_2d, dim=-1)
        index = soft.argmax(dim=-1)
        hard = F.one_hot(index, self.stoch_classes).float()
        z = hard + soft - soft.detach()
        return z.view(B, self.stoch_flat_dim)

    def observe_step(self, prev_h, prev_z, action, obs_embed):
        gru_input = torch.cat([prev_z, action], dim=-1)
        h_t = self.gru(gru_input, prev_h)

        post_input = torch.cat([h_t, obs_embed], dim=-1)
        post_logits = self.posterior_net(post_input)
        z_t = self._sample_straight_through(post_logits)

        prior_logits = self.prior_net(h_t)

        return h_t, z_t, prior_logits, post_logits

    def imagine_step(self, prev_h, prev_z, action):
        gru_input = torch.cat([prev_z, action], dim=-1)
        h_t = self.gru(gru_input, prev_h)

        prior_logits = self.prior_net(h_t)
        z_t = self._sample_straight_through(prior_logits)

        s_t = torch.cat([h_t, z_t], dim=-1)
        reward_symlog = torch.clamp(self.reward_head(s_t), -20.0, 20.0)
        continue_logit = self.continue_head(s_t)

        return h_t, z_t, prior_logits, reward_symlog, continue_logit

    def get_state_features(self, h, z):
        return torch.cat([h, z], dim=-1)

    @staticmethod
    def kl_loss(prior_logits, post_logits, stoch_dim, stoch_classes,
                free_bits=1.0, kl_balance=0.8):
        B = prior_logits.shape[0]
        prior_logits_2d = prior_logits.view(B, stoch_dim, stoch_classes)
        post_logits_2d = post_logits.view(B, stoch_dim, stoch_classes)

        prior_dist = Categorical(logits=prior_logits_2d)
        post_dist = Categorical(logits=post_logits_2d)

        kl_dyn = torch.distributions.kl_divergence(
            Categorical(logits=post_logits_2d.detach()),
            prior_dist
        ).clamp(min=free_bits).mean()

        kl_rep = torch.distributions.kl_divergence(
            post_dist,
            Categorical(logits=prior_logits_2d.detach())
        ).clamp(min=free_bits).mean()

        kl_balanced = kl_balance * kl_dyn + (1.0 - kl_balance) * kl_rep

        kl_raw = torch.distributions.kl_divergence(
            post_dist, prior_dist
        ).mean().item()

        return kl_balanced, kl_raw


class MuDreamerActorCritic(nn.Module):
    # Bounds on the actor's Gaussian log_std. -20 (original) and -5 (first
    # fix attempt) both still allowed the policy to collapse: at -5,
    # std ~= 0.0067, and with a 4-dim action space the summed Normal
    # log-density near the mean can reach ~16 — multiplied by an advantage
    # of order ~1 this lands almost exactly on the -14ish actor loss ceiling
    # observed in every collapse so far, independent of *when* it happens.
    # -3 (std ~= 0.05) cuts that log-density ceiling roughly in half and
    # still leaves room for a fairly confident policy once training is
    # actually going well. Paired with the advantage clipping below (which
    # bounds the OTHER multiplicand in the same product), this removes both
    # halves of the mechanism that produced the collapse.
    LOG_STD_MIN = -3.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim, action_dim, conditioning_dim=0):
        super().__init__()
        actor_input_dim = state_dim + conditioning_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, action_dim * 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def get_action_dist(self, state, conditioning=None):
        actor_input = (
            torch.cat([state, conditioning], dim=-1)
            if conditioning is not None
            else state
        )
        stats = self.actor(actor_input)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.distributions.Normal(mean, torch.exp(log_std))

    def get_action(self, state, conditioning=None, deterministic=False):
        dist = self.get_action_dist(state, conditioning)
        return torch.tanh(dist.mean if deterministic else dist.rsample())

    def get_value(self, state):
        raw = torch.clamp(self.critic(state), -10.0, 10.0)
        return symexp(raw)


class MuDreamerWorldModelTrainer:
    def __init__(self, rssm, lr=1e-4, grad_clip=10.0, kl_weight=1.0):
        self.rssm = rssm
        self.optimizer = optim.Adam(self.rssm.parameters(), lr=lr)
        self.grad_clip = grad_clip
        self.kl_weight = kl_weight

    def train_step_sequence(self, obs_seq, actions_seq, rewards_seq, dones_seq):
        """
        Train RSSM on a batch of contiguous sequences.

        obs_seq:     (B, L, obs_embed_dim)
        actions_seq: (B, L, action_dim)
        rewards_seq: (B, L)
        dones_seq:   (B, L)

        The GRU hidden state h is propagated through the full sequence length L
        so the model learns to use temporal context. KL is now meaningful because
        h_t carries real context from previous steps rather than always being zero.
        """
        self.optimizer.zero_grad()

        B, L, _ = obs_seq.shape
        device = obs_seq.device

        prev_h, prev_z = self.rssm.initial_state(B, device)
        prev_action = torch.zeros(B, actions_seq.shape[-1], device=device)

        total_latent_loss = torch.tensor(0.0, device=device)
        total_reward_loss = torch.tensor(0.0, device=device)
        total_continue_loss = torch.tensor(0.0, device=device)
        total_kl = torch.tensor(0.0, device=device)
        total_kl_raw = 0.0

        # posterior_net has BatchNorm1d — must be in train mode with B > 1
        self.rssm.posterior_net.train()

        for t in range(L):
            obs_t = obs_seq[:, t, :]
            action_t = actions_seq[:, t, :]
            reward_t = rewards_seq[:, t].unsqueeze(-1)
            done_t = dones_seq[:, t].unsqueeze(-1)

            h_t, z_t, prior_logits, post_logits = self.rssm.observe_step(
                prev_h, prev_z, prev_action, obs_t
            )
            s_t = self.rssm.get_state_features(h_t, z_t)

            # Latent prediction: prior should predict what posterior produced
            prior_z_pred = self.rssm._sample_straight_through(prior_logits)
            latent_loss = torch.mean((prior_z_pred - z_t.detach()) ** 2)
            total_latent_loss = total_latent_loss + latent_loss

            # Reward prediction in symlog space
            pred_reward_symlog = torch.clamp(
                self.rssm.reward_head(s_t), -20.0, 20.0
            )
            reward_loss = torch.mean(
                (pred_reward_symlog - symlog(reward_t)) ** 2
            )
            total_reward_loss = total_reward_loss + reward_loss

            # Continue prediction
            continue_target = 1.0 - done_t
            pred_continue_logit = self.rssm.continue_head(s_t)
            continue_loss = nn.functional.binary_cross_entropy_with_logits(
                pred_continue_logit, continue_target
            )
            total_continue_loss = total_continue_loss + continue_loss

            # KL loss — meaningful because h_t carries sequence context
            kl, kl_raw = RSSMCore.kl_loss(
                prior_logits, post_logits,
                self.rssm.stoch_dim, self.rssm.stoch_classes,
            )
            total_kl = total_kl + kl
            total_kl_raw += kl_raw

            # Carry hidden state forward; reset at episode boundaries
            reset_mask = 1.0 - done_t  # (B, 1)
            prev_h = h_t * reset_mask
            prev_z = z_t * reset_mask
            prev_action = action_t

        loss = (
            total_latent_loss / L
            + total_reward_loss / L
            + total_continue_loss / L
            + self.kl_weight * total_kl / L
        )

        loss.backward()
        nn.utils.clip_grad_norm_(self.rssm.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "world_model_loss": loss.item(),
            "latent_loss": (total_latent_loss / L).item(),
            "reward_loss": (total_reward_loss / L).item(),
            "continue_loss": (total_continue_loss / L).item(),
            "kl_loss": (total_kl / L).item(),
            "kl_raw": total_kl_raw / L,
        }


class MuDreamerActorTrainer:
    # After normalization, advantage is hard-clamped to this range. The
    # previous scheme (divide by clamp(mean(abs(advantage)), min=1.0)) only
    # sets a FLOOR on the scale estimate — it does nothing to stop a single
    # large outlier advantage from surviving the normalization and then
    # multiplying with a near-collapsed log_std's huge log-density (see
    # LOG_STD_MIN comment above) to produce a runaway policy_loss. A hard
    # clip after normalization bounds that product's magnitude unconditionally,
    # regardless of how good or bad the scale estimate turns out to be for a
    # given batch.
    ADVANTAGE_CLIP = 5.0

    def __init__(
        self, actor_critic, lr=3e-5, entropy_scale=3e-4, grad_clip=10.0,
        auxiliary_params=None, auxiliary_lr_scale=0.5, action_mean_penalty=1e-3,
    ):
        """
        auxiliary_params: optional iterable of extra parameters (e.g. a BTN
        module's parameters()) to include in THIS optimizer's step. This is
        the fix for a real bug: the BTN is called inside the imagination
        rollout (feeding conditioning into the actor), so backward() through
        the actor loss genuinely computes gradients on the BTN's parameters
        — but previously nothing ever applied them, because the actor
        optimizer only covered actor.parameters(), and BTNTrainer's own
        optimizer.zero_grad() (called separately, for its own entropy-only
        loss) wiped those gradients before they were ever used. Given as a
        separate, lower-lr param group by default (auxiliary_lr_scale=0.5)
        since the policy gradient signal is noisier than a dedicated
        supervised loss.

        action_mean_penalty: weight on an L2 penalty over the actor's RAW
        (pre-tanh) mean output. Diagnostic evidence from a 150k-step run:
        FracSaturated (fraction of action dims with |tanh(action)| > 0.99)
        locked at EXACTLY 1.00 for the final 130000 steps (86% of the run),
        with MeanAbsAction ~0.9998 and Mean Final Distance frozen bit-
        identically the entire time. Nothing in the loss up to this point
        discouraged the pre-tanh mean from growing without bound, and once
        it grows large enough tanh saturates completely — at which point
        tanh's own gradient is ~0, so there is essentially no way for
        further policy-gradient steps to ever pull it back down; the
        network gets stuck by construction, not by insufficient training.
        This penalty gives the loss an explicit, ever-present incentive to
        keep the pre-tanh mean bounded, so saturation only happens when the
        policy is actually confident (which is fine) rather than growing
        unboundedly with no counter-pressure.
        """
        self.ac = actor_critic
        self._btn_params = list(auxiliary_params) if auxiliary_params is not None else []
        param_groups = [{"params": list(self.ac.actor.parameters()), "lr": lr}]
        if self._btn_params:
            param_groups.append({"params": self._btn_params, "lr": lr * auxiliary_lr_scale})
        self.optimizer = optim.Adam(param_groups)
        self.entropy_scale = entropy_scale
        self.grad_clip = grad_clip
        self.action_mean_penalty = action_mean_penalty

    def train_step(
        self, imagined_states, imagined_log_probs, imagined_entropy,
        lambda_returns, values, imagined_conditioning=None,
        imagined_raw_means=None,
    ):
        self.optimizer.zero_grad()

        advantage = (lambda_returns - values).detach()
        # Scale by standard deviation rather than mean absolute value — std
        # reflects the spread of the whole batch, so a handful of outliers
        # can't single-handedly shrink the effective scale the way a mean
        # can be dragged around by them. Floor still applied so a
        # near-constant advantage batch doesn't get divided by ~0.
        adv_scale = torch.clamp(advantage.std(), min=1.0)
        advantage = advantage / adv_scale
        # Hard clip regardless of scale estimate quality — see ADVANTAGE_CLIP.
        advantage = torch.clamp(advantage, -self.ADVANTAGE_CLIP, self.ADVANTAGE_CLIP)

        policy_loss = -(imagined_log_probs * advantage).mean()
        entropy_loss = -self.entropy_scale * imagined_entropy.mean()

        mean_penalty_loss = torch.tensor(0.0, device=advantage.device)
        if imagined_raw_means is not None and self.action_mean_penalty > 0:
            mean_penalty_loss = self.action_mean_penalty * (imagined_raw_means ** 2).mean()

        loss = policy_loss + entropy_loss + mean_penalty_loss
        loss.backward()
        # Clip over BOTH param groups — actor and (if present) BTN — so the
        # newly-connected BTN gradient can't produce a runaway update any
        # more than the actor's own gradient can.
        clip_params = list(self.ac.actor.parameters()) + self._btn_params
        nn.utils.clip_grad_norm_(clip_params, self.grad_clip)
        self.optimizer.step()

        return {
            "actor_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_penalty_loss": mean_penalty_loss.item(),
        }


class MuDreamerValueTrainer:
    """
    Critic trained in symlog space with slow EMA target network.
    Correct TD(lambda) — lambda is not algebraically cancelled.
    """
    def __init__(
        self,
        critic_net,
        lr=1e-4,
        discount=0.99,
        lambda_=0.95,
        target_ema_decay=0.98,
        grad_clip=10.0,
    ):
        self.critic = critic_net
        self.optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.discount = discount
        self.lambda_ = lambda_
        self.grad_clip = grad_clip
        self.target_ema_decay = target_ema_decay

        self.target_critic = copy.deepcopy(critic_net)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.target_critic.eval()

    @torch.no_grad()
    def _update_target(self):
        for mp, tp in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            tp.data.mul_(self.target_ema_decay).add_(
                mp.data, alpha=1.0 - self.target_ema_decay
            )

    def compute_lambda_returns(
        self, rewards_symlog, continues, all_target_values_symlog
    ):
        """
        Correct TD(lambda):
        G_H = V_target(s_H)
        G_t = r_t + gamma*c_t * [(1-lambda)*V_target(s_{t+1}) + lambda*G_{t+1}]
        """
        H = rewards_symlog.size(0)
        returns = torch.zeros_like(rewards_symlog)
        G = all_target_values_symlog[H]

        for t in reversed(range(H)):
            V_next = all_target_values_symlog[t + 1]
            G = rewards_symlog[t] + self.discount * continues[t] * (
                (1 - self.lambda_) * V_next + self.lambda_ * G
            )
            returns[t] = G

        return returns

    def train_step(self, imagined_states, imagined_rewards_symlog, imagined_continues):
        self.optimizer.zero_grad()

        H_plus_1, B, D = imagined_states.shape
        flat_states = imagined_states.view(-1, D)

        all_values_symlog = torch.clamp(
            self.critic(flat_states), -10.0, 10.0
        ).view(H_plus_1, B)
        trajectory_values_symlog = all_values_symlog[:-1]

        with torch.no_grad():
            all_target_values_symlog = torch.clamp(
                self.target_critic(flat_states), -10.0, 10.0
            ).view(H_plus_1, B)
            targets_symlog = self.compute_lambda_returns(
                imagined_rewards_symlog,
                imagined_continues,
                all_target_values_symlog,
            )

        loss = 0.5 * torch.mean(
            (trajectory_values_symlog - targets_symlog) ** 2
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.optimizer.step()
        self._update_target()

        return (
            loss.item(),
            symexp(targets_symlog).detach(),
            symexp(trajectory_values_symlog).detach(),
        )