import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim

class MuDreamerPredictor(nn.Module):
    """
    Transition model over the latent space.
    Predicts next latent state AND reward/continue from current state and action.
    """
    def __init__(self, embed_dim=512, action_dim=6):
        super().__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 512),
            nn.ELU(),
            nn.Linear(512, embed_dim)
        )
        # Reward head: predicts reward of the transition into next_latent
        self.reward_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        # Continue head: predicts probability episode continues (not done)
        self.continue_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        next_latent = self.dynamics(torch.cat([state, action], dim=-1))
        reward = self.reward_head(next_latent)
        continue_logit = self.continue_head(next_latent)
        return next_latent, reward, continue_logit


class MuDreamerActorCritic(nn.Module):
    """
    Policy engine that maps imagined latents directly to continuous actions.
    """
    def __init__(self, embed_dim=512, action_dim=6):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, action_dim * 2)  # Mean and LogStd
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def get_action_dist(self, state):
        stats = self.actor(state)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_action(self, state, deterministic=False):
        dist = self.get_action_dist(state)
        if deterministic:
            return torch.tanh(dist.mean)
        return torch.tanh(dist.rsample())

    def get_value(self, state):
        return self.critic(state)


class MuDreamerWorldModelTrainer:
    """
    Trains the predictor's dynamics/reward/continue heads using real
    (state, action, next_state, reward, done) transitions from the replay buffer.
    This is the missing Lpred / world-model loss.
    """
    def __init__(self, predictor, lr=3e-4):
        self.predictor = predictor
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

    def train_step(self, states, actions, next_states, rewards, dones):
        """
        states, next_states: (B, embed_dim)
        actions: (B, action_dim)
        rewards: (B, 1)
        dones: (B, 1)  -- 1.0 if episode ended, else 0.0
        """
        self.optimizer.zero_grad()

        pred_next_latent, pred_reward, pred_continue_logit = self.predictor(states, actions)

        latent_loss = torch.mean((pred_next_latent - next_states.detach()) ** 2)
        reward_loss = torch.mean((pred_reward - rewards) ** 2)
        continue_target = 1.0 - dones
        continue_loss = nn.functional.binary_cross_entropy_with_logits(
            pred_continue_logit, continue_target
        )

        loss = latent_loss + reward_loss + continue_loss
        loss.backward()
        self.optimizer.step()

        return {
            "world_model_loss": loss.item(),
            "latent_loss": latent_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
        }


class MuDreamerActorTrainer:
    """
    Trains the actor using lambda-returns computed from imagined rollouts,
    with an entropy bonus for exploration. This was completely missing before.
    """
    def __init__(self, actor_critic, lr=1e-4, entropy_scale=3e-4):
        self.ac = actor_critic
        self.optimizer = optim.Adam(self.ac.actor.parameters(), lr=lr)
        self.entropy_scale = entropy_scale

    def train_step(self, imagined_states, imagined_log_probs, imagined_entropy, lambda_returns, values):
        """
        imagined_states: (H, B, D) - states the actor acted from
        imagined_log_probs: (H, B) - log prob of actions taken at each step
        imagined_entropy: (H, B) - entropy of action dist at each step
        lambda_returns: (H, B) - targets from critic trainer
        values: (H, B) - critic's value estimate at each step (detached baseline)
        """
        self.optimizer.zero_grad()

        advantage = (lambda_returns - values).detach()
        # Reinforce-style policy gradient (works for tanh-squashed Normal too,
        # since we are not backpropping through the dynamics here for stability)
        policy_loss = -(imagined_log_probs * advantage).mean()
        entropy_loss = -self.entropy_scale * imagined_entropy.mean()

        loss = policy_loss + entropy_loss
        loss.backward()
        self.optimizer.step()

        return {"actor_loss": loss.item(), "policy_loss": policy_loss.item()}


class MuDreamerValueTrainer:
    """
    Value Network update utilizing Lambda-returns over PREDICTED imagined rewards
    (not a broadcast scalar).
    """
    def __init__(self, critic_net, lr=3e-4, discount=0.99, lambda_=0.95):
        self.critic = critic_net
        self.optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.discount = discount
        self.lambda_ = lambda_

    def compute_lambda_returns(self, rewards, values, continues, baseline_value):
        """
        rewards: (H, B)
        values: (H, B)
        continues: (H, B) -- predicted continue probability (0..1) per step
        baseline_value: (B,)
        """
        horizon = rewards.size(0)
        next_values = torch.cat([values[1:], baseline_value.unsqueeze(0)], dim=0)

        returns = torch.zeros_like(rewards)
        last_lambda_return = baseline_value

        for t in reversed(range(horizon)):
            td_target = rewards[t] + self.discount * continues[t] * next_values[t]
            returns[t] = td_target + self.discount * continues[t] * self.lambda_ * (
                last_lambda_return - next_values[t]
            )
            last_lambda_return = returns[t]

        return returns

    def train_step(self, imagined_latents, imagined_rewards, imagined_continues):
        """
        imagined_latents: (H+1, B, D)
        imagined_rewards: (H, B)   -- predicted, not broadcast real reward
        imagined_continues: (H, B) -- predicted continue probabilities
        """
        self.optimizer.zero_grad()

        H_plus_1, B, D = imagined_latents.shape
        flat_latents = imagined_latents.view(-1, D)
        all_values = self.critic(flat_latents).view(H_plus_1, B)

        trajectory_values = all_values[:-1]
        final_step_value = all_values[-1].detach()

        targets = self.compute_lambda_returns(
            rewards=imagined_rewards,
            values=trajectory_values.detach(),
            continues=imagined_continues,
            baseline_value=final_step_value,
        )

        loss = 0.5 * torch.mean((trajectory_values - targets) ** 2)
        loss.backward()
        self.optimizer.step()
        return loss.item(), targets.detach(), trajectory_values.detach()