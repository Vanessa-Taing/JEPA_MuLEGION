import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import copy


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    # Clamp before expm1 — float32 overflows above ~88, producing inf
    # which then propagates as NaN through inf-inf in advantage computation
    x = torch.clamp(x, min=-20.0, max=20.0)
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


class MuDreamerPredictor(nn.Module):
    """
    World model transition network with GRU sequential memory.

    The GRU gives the model memory across the imagination horizon —
    without it, every imagined step is conditioned only on the current
    latent and action, making it impossible to represent approach
    trajectories or multi-step sub-goals (per MuDreamer paper Section 4.1).

    During real-transition world-model training (random buffer batches),
    hidden state is initialized to zeros since we don't store sequences.
    During imagination rollouts, hidden state is carried across steps.
    """
    def __init__(self, embed_dim=512, action_dim=6, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # GRU sequential network: gives temporal memory across horizon
        self.gru = nn.GRUCell(embed_dim + action_dim, hidden_dim)

        # Dynamics: predict next latent from GRU hidden state
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim)
        )

        # Reward and continue heads read from GRU hidden state
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        self.continue_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action, hidden=None):
        """
        state:  (B, embed_dim)
        action: (B, action_dim)
        hidden: (B, hidden_dim) or None — GRU hidden state

        Returns next_latent, reward_symlog, continue_logit, hidden
        """
        B = state.shape[0]
        device = state.device

        if hidden is None:
            hidden = torch.zeros(B, self.hidden_dim, device=device)

        gru_input = torch.cat([state, action], dim=-1)
        hidden = self.gru(gru_input, hidden)

        next_latent = self.dynamics(hidden)
        # Clamp in symlog space — predictor output feeds directly into
        # lambda-return computation, so early instability must be caught here
        reward_symlog = torch.clamp(
            self.reward_head(hidden), min=-20.0, max=20.0
        )
        continue_logit = self.continue_head(hidden)

        return next_latent, reward_symlog, continue_logit, hidden


class MuDreamerActorCritic(nn.Module):
    """
    Policy engine. Actor optionally accepts semantic conditioning from the BTN.
    Critic operates on pure latent state (value is a dynamics property, not semantic).
    """
    def __init__(self, embed_dim=512, action_dim=6, conditioning_dim=0):
        super().__init__()
        # conditioning_dim=0 means no BTN — backward compatible
        actor_input_dim = embed_dim + conditioning_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, 256),
            nn.ELU(),
            nn.Linear(256, action_dim * 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def get_action_dist(self, state, conditioning=None):
        if conditioning is not None:
            actor_input = torch.cat([state, conditioning], dim=-1)
        else:
            actor_input = state
        stats = self.actor(actor_input)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_action(self, state, conditioning=None, deterministic=False):
        dist = self.get_action_dist(state, conditioning)
        if deterministic:
            return torch.tanh(dist.mean)
        return torch.tanh(dist.rsample())

    def get_value(self, state):
        # Clamp raw critic output in symlog space before symexp
        raw = torch.clamp(self.critic(state), -10.0, 10.0)
        return symexp(raw)


class MuDreamerWorldModelTrainer:
    def __init__(self, predictor, lr=3e-4, grad_clip=10.0):
        self.predictor = predictor
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.grad_clip = grad_clip

    def train_step(self, states, actions, next_states, rewards, dones):
        """
        Random buffer batches — GRU hidden initialized to zeros since
        we do not store hidden states alongside transitions.
        This is a known approximation; sequence-aware replay is a future fix.
        """
        self.optimizer.zero_grad()

        B = states.shape[0]
        hidden = torch.zeros(B, self.predictor.hidden_dim, device=states.device)

        pred_next_latent, pred_reward_symlog, pred_continue_logit, _ = self.predictor(
            states, actions, hidden
        )

        latent_loss = torch.mean((pred_next_latent - next_states.detach()) ** 2)
        reward_target_symlog = symlog(rewards)
        reward_loss = torch.mean((pred_reward_symlog - reward_target_symlog) ** 2)
        continue_target = 1.0 - dones
        continue_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_continue_logit, continue_target
        )

        loss = latent_loss + reward_loss + continue_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "world_model_loss": loss.item(),
            "latent_loss": latent_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
        }


class MuDreamerActorTrainer:
    def __init__(self, actor_critic, lr=1e-4, entropy_scale=3e-4, grad_clip=10.0):
        self.ac = actor_critic
        self.optimizer = optim.Adam(self.ac.actor.parameters(), lr=lr)
        self.entropy_scale = entropy_scale
        self.grad_clip = grad_clip

    def train_step(
        self, imagined_states, imagined_log_probs, imagined_entropy,
        lambda_returns, values, imagined_conditioning=None
    ):
        self.optimizer.zero_grad()

        advantage = (lambda_returns - values).detach()
        # Normalize advantage — prevents extreme policy gradient steps when
        # value estimates are still inaccurate early in training
        adv_scale = torch.clamp(advantage.abs().mean(), min=1.0)
        advantage = advantage / adv_scale

        policy_loss = -(imagined_log_probs * advantage).mean()
        entropy_loss = -self.entropy_scale * imagined_entropy.mean()

        loss = policy_loss + entropy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac.actor.parameters(), self.grad_clip)
        self.optimizer.step()

        return {"actor_loss": loss.item(), "policy_loss": policy_loss.item()}


class MuDreamerValueTrainer:
    """
    Critic trained entirely in symlog space with a slow EMA target network.

    Key fix: compute_lambda_returns now uses target critic values at EVERY
    intermediate step, not just the final bootstrap — this is the correct
    TD(lambda) formula. The previous version computed
    (1-lambda)*x + lambda*x = x, making lambda a no-op.
    """
    def __init__(
        self,
        critic_net,
        lr=3e-4,
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
        for main_p, target_p in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_p.data.mul_(self.target_ema_decay).add_(
                main_p.data, alpha=1.0 - self.target_ema_decay
            )

    def compute_lambda_returns(
        self, rewards_symlog, continues, all_target_values_symlog
    ):
        """
        Correct TD(lambda) backward pass.

        rewards_symlog:           (H, B)   predictor reward output
        continues:                (H, B)   predicted episode-continue probability
        all_target_values_symlog: (H+1, B) target critic at every imagined state

        G_H = V_target(s_H)
        G_t = r_t + gamma*c_t * [(1-lambda)*V_target(s_{t+1}) + lambda*G_{t+1}]

        The (1-lambda) term blends toward the 1-step TD target at each step,
        providing variance reduction vs pure Monte Carlo. Previously this term
        was algebraically cancelled out.
        """
        H = rewards_symlog.size(0)
        returns = torch.zeros_like(rewards_symlog)

        # Bootstrap from the target critic at the final imagined state
        G = all_target_values_symlog[H]  # (B,)

        for t in reversed(range(H)):
            V_next = all_target_values_symlog[t + 1]  # target value at s_{t+1}
            G = rewards_symlog[t] + self.discount * continues[t] * (
                (1 - self.lambda_) * V_next + self.lambda_ * G
            )
            returns[t] = G

        return returns

    def train_step(self, imagined_latents, imagined_rewards_symlog, imagined_continues):
        self.optimizer.zero_grad()

        H_plus_1, B, D = imagined_latents.shape
        flat_latents = imagined_latents.view(-1, D)

        # Main critic — receives gradients, output clamped in symlog space
        all_values_symlog = torch.clamp(
            self.critic(flat_latents), -10.0, 10.0
        ).view(H_plus_1, B)
        trajectory_values_symlog = all_values_symlog[:-1]  # (H, B)

        with torch.no_grad():
            # Target critic — stable bootstrap values, no gradients
            all_target_values_symlog = torch.clamp(
                self.target_critic(flat_latents), -10.0, 10.0
            ).view(H_plus_1, B)

            targets_symlog = self.compute_lambda_returns(
                rewards_symlog=imagined_rewards_symlog,
                continues=imagined_continues,
                all_target_values_symlog=all_target_values_symlog,
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