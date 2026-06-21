import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim

class MuDreamerPredictor(nn.Module):
    """
    MuDreamer's transition model over the regularized LeWM latent space.
    Predicts the next latent state purely from current state and action.
    """
    def __init__(self, embed_dim=512, action_dim=6):
        super().__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 512),
            nn.ELU(),
            nn.Linear(512, embed_dim)
        )
        
    def forward(self, state, action):
        # Non-reconstructive forward projection step
        return self.dynamics(torch.cat([state, action], dim=-1))

class MuDreamerActorCritic(nn.Module):
    """
    MuDreamer policy engine that maps imagined latents directly to continuous actions.
    """
    def __init__(self, embed_dim=512, action_dim=6):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, action_dim * 2) # Mean and LogStd
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
        
    def get_action(self, state, deterministic=False):
        stats = self.actor(state)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        if deterministic:
            return torch.tanh(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return torch.tanh(dist.rsample())

    def get_value(self, state):
        return self.critic(state)

class CombinedJEPAWorldModel(nn.Module):
    """
    The unified architecture combining LeWM representations with MuDreamer policy.
    """
    def __init__(self, action_dim=6, embed_dim=512):
        super().__init__()
        self.encoder = LeWMEncoder(embed_dim=embed_dim)
        self.predictor = MuDreamerPredictor(embed_dim=embed_dim, action_dim=action_dim)
        self.ac = MuDreamerActorCritic(embed_dim=embed_dim, action_dim=action_dim)
        
    def imagine_trajectory(self, initial_pixels, horizon=15):
        """
        Rolls out imaginary vectors forward in time to train the policy.
        Bypasses pixel-space decoders entirely.
        """
        B = initial_pixels.size(0)
        device = initial_pixels.device
        
        # 1. Encode initial visual observation via LeWM
        current_latent = self.encoder(initial_pixels)
        
        imagined_latents = [current_latent]
        imagined_actions = []
        
        # 2. Project forward in abstract latent space (MuDreamer Imagination Loop)
        for _ in range(horizon):
            action = self.ac.get_action(current_latent)
            next_latent = self.predictor(current_latent, action)
            
            imagined_latents.append(next_latent)
            imagined_actions.append(action)
            current_latent = next_latent
            
        return torch.stack(imagined_latents), torch.stack(imagined_actions)

class MuDreamerValueTrainer:
    """
    Implements a non-reconstructive Value Network update utilizing 
    Lambda-returns over normalized KUKA multi-task rewards.
    """
    def __init__(self, critic_net, lr=3e-4, discount=0.99, lambda_=0.95):
        self.critic = critic_net
        self.optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.discount = discount
        self.lambda_ = lambda_

    def compute_lambda_returns(self, rewards, values, baseline_value):
        """
        Calculates targets across the imagined trajectory horizon.
        rewards shape: (H, B)
        values shape: (H, B)
        baseline_value shape: (B,) - Final step bootstrap value
        """
        horizon = rewards.size(0)
        batch_size = rewards.size(1)
        
        # Append the final terminal bootstrap value to make the array size H+1
        next_values = torch.cat([values[1:], baseline_value.unsqueeze(0)], dim=0)
        
        # Compute multi-step TD targets backwards through time
        returns = torch.zeros_like(rewards)
        last_lambda_return = baseline_value
        
        for t in reversed(range(horizon)):
            # TD(0) target component
            td_target = rewards[t] + self.discount * next_values[t]
            # Mix with multi-step lambda lookahead
            returns[t] = td_target + self.discount * self.lambda_ * (last_lambda_return - next_values[t])
            last_lambda_return = returns[t]
            
        return returns

    def train_step(self, imagined_latents, normalized_rewards):
        """
        Performs gradient descent backpropagation over the value heads.
        imagined_latents: (H + 1, B, Embed_Dim) -> Generated from MuDreamer predictor
        normalized_rewards: (H, B)             -> Filtered via LegionRewardNormalizer
        """
        self.optimizer.zero_grad()
        
        # Forward pass for all steps across the horizon
        H_plus_1, B, D = imagined_latents.shape
        flat_latents = imagined_latents.view(-1, D)
        all_values = self.critic(flat_latents).view(H_plus_1, B)
        
        # Split into horizon track steps and the final boundary element
        trajectory_values = all_values[:-1]
        final_step_value = all_values[-1].detach()
        
        # Target estimation loop
        targets = self.compute_lambda_returns(
            rewards=normalized_rewards, 
            values=trajectory_values.detach(), 
            baseline_value=final_step_value
        )
        
        # Compute mean squared error regression loss
        loss = 0.5 * torch.mean((trajectory_values - targets) ** 2)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
