import numpy as np
import torch
from collections import deque

import numpy as np
import torch
from collections import deque


class DirichletProcessMixtureAllocator:
    """
    Online Dirichlet Process Mixture Model for task/skill clustering.

    Differences from the previous hard-assignment allocator:
    - Each cluster k is a full Gaussian N(mu_k, Sigma_k) (diagonal covariance)
      instead of just a centroid.
    - Assignment is SOFT: p(v=k | z) computed via Gaussian log-likelihood + softmax,
      mirroring the p_{t,k} formulation in the Tensor-MuLEGION design docs.
    - A "birth" move triggers when the max soft-assignment probability across all
      existing clusters falls below `alpha` (the concentration/surprise threshold),
      analogous to the DPMM birth condition described in your architecture notes.
    - A "merge" move periodically collapses two clusters whose means are very close,
      preventing unbounded cluster growth from noisy assignments.
    """

    def __init__(
        self,
        latent_dim=512,
        alpha=0.3,           # concentration / birth threshold on max soft-assignment prob
        max_components=50,
        warmup_steps=50,
        ema_alpha=0.05,       # mean/var update rate for the assigned cluster
        merge_check_interval=500,
        merge_threshold=0.05,  # cosine distance below which two clusters get merged
        init_var=1.0,
    ):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.max_components = max_components
        self.warmup_steps = warmup_steps
        self.ema_alpha = ema_alpha
        self.merge_check_interval = merge_check_interval
        self.merge_threshold = merge_threshold
        self.init_var = init_var

        # Each entry: {"mean": np.array(D,), "var": np.array(D,), "count": int}
        self.component_profiles = {}

    # ---------- helpers ----------

    def normalize_latent(self, latent):
        norm = np.linalg.norm(latent)
        if norm < 1e-8:
            return latent
        return latent / norm

    def _gaussian_log_likelihood(self, z, mean, var):
        """Diagonal Gaussian log-likelihood, summed over dimensions."""
        var = np.clip(var, 1e-4, None)
        log_prob = -0.5 * np.sum(
            np.log(2 * np.pi * var) + ((z - mean) ** 2) / var
        )
        return log_prob

    def _soft_assignments(self, z):
        """
        Returns dict {cluster_id: probability} via softmax over Gaussian
        log-likelihoods across all active clusters — the p_{t,k} in the design docs.
        """
        if not self.component_profiles:
            return {}

        ids = list(self.component_profiles.keys())
        log_probs = np.array([
            self._gaussian_log_likelihood(
                z,
                self.component_profiles[cid]["mean"],
                self.component_profiles[cid]["var"],
            )
            for cid in ids
        ])

        # softmax for numerical stability
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= (np.sum(probs) + 1e-12)

        return dict(zip(ids, probs))

    def _update_component(self, cid, z):
        """EMA update of mean and variance for the assigned component (online memoVB-style)."""
        profile = self.component_profiles[cid]
        old_mean = profile["mean"]

        new_mean = (1 - self.ema_alpha) * old_mean + self.ema_alpha * z
        # online variance update (EMA of squared deviation)
        deviation_sq = (z - new_mean) ** 2
        new_var = (1 - self.ema_alpha) * profile["var"] + self.ema_alpha * deviation_sq

        profile["mean"] = new_mean
        profile["var"] = np.maximum(new_var, 1e-4)
        profile["count"] += 1

    def _maybe_merge(self, step_idx, logger):
        """Birth/merge balance: collapse near-duplicate clusters created from noisy assignments."""
        if len(self.component_profiles) < 2:
            return
        if step_idx % self.merge_check_interval != 0:
            return

        ids = list(self.component_profiles.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cid_a, cid_b = ids[i], ids[j]
                if cid_a not in self.component_profiles or cid_b not in self.component_profiles:
                    continue
                mean_a = self.component_profiles[cid_a]["mean"]
                mean_b = self.component_profiles[cid_b]["mean"]

                cos_sim = np.dot(
                    self.normalize_latent(mean_a), self.normalize_latent(mean_b)
                )
                cos_dist = 1.0 - cos_sim

                if cos_dist < self.merge_threshold:
                    # merge b into a, weighted by counts
                    count_a = self.component_profiles[cid_a]["count"]
                    count_b = self.component_profiles[cid_b]["count"]
                    total = count_a + count_b

                    merged_mean = (mean_a * count_a + mean_b * count_b) / total
                    merged_var = (
                        self.component_profiles[cid_a]["var"] * count_a
                        + self.component_profiles[cid_b]["var"] * count_b
                    ) / total

                    self.component_profiles[cid_a]["mean"] = merged_mean
                    self.component_profiles[cid_a]["var"] = merged_var
                    self.component_profiles[cid_a]["count"] = total

                    del self.component_profiles[cid_b]

                    if logger is not None:
                        logger.log_task_expansion(
                            step_idx,
                            active_task_id=cid_a,
                            new_total_components=len(self.component_profiles),
                            similarity_score=cos_sim,
                        )

    # ---------- main API (same signature as before) ----------

    def evaluate_task_assignment(self, current_latent_tensor, step_idx, logger):
        z = (
            current_latent_tensor
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )
        z = self.normalize_latent(z)

        # First cluster ever
        if len(self.component_profiles) == 0:
            self.component_profiles[0] = {
                "mean": z.copy(),
                "var": np.full_like(z, self.init_var),
                "count": 1,
            }
            if logger is not None:
                logger.log_task_expansion(
                    step_idx, active_task_id=0, new_total_components=1, similarity_score=1.0
                )
            return 0, True

        soft_probs = self._soft_assignments(z)
        best_id = max(soft_probs, key=soft_probs.get)
        best_prob = soft_probs[best_id]

        # Birth condition: max soft-assignment probability below alpha threshold
        # i.e. the latent doesn't fit any existing cluster well (surprise > alpha)
        should_birth = (
            best_prob < self.alpha
            and step_idx >= self.warmup_steps
            and len(self.component_profiles) < self.max_components
        )

        if should_birth:
            new_id = max(self.component_profiles.keys()) + 1
            self.component_profiles[new_id] = {
                "mean": z.copy(),
                "var": np.full_like(z, self.init_var),
                "count": 1,
            }
            if logger is not None:
                logger.log_task_expansion(
                    step_idx,
                    active_task_id=new_id,
                    new_total_components=len(self.component_profiles),
                    similarity_score=best_prob,
                )
            self._maybe_merge(step_idx, logger)
            return new_id, True

        # Otherwise, assign to best-matching cluster and update it
        self._update_component(best_id, z)
        self._maybe_merge(step_idx, logger)
        return best_id, False


# Backward-compatible alias so existing imports of LegionComponentAllocator still work
LegionComponentAllocator = DirichletProcessMixtureAllocator

# ==============================================================
# Reward Normalizer
# ==============================================================


class LegionRewardNormalizer:


    def __init__(
        self,
        clip_range=(-5,5),
        momentum=0.99
    ):

        self.mean=0
        self.var=1
        self.momentum=momentum



    def update_statistics(
        self,
        rewards
    ):

        rewards=np.array(rewards)


        batch_mean=rewards.mean()

        batch_var=rewards.var()


        self.mean=(
            self.momentum*self.mean
            +
            (1-self.momentum)*batch_mean
        )


        self.var=(
            self.momentum*self.var
            +
            (1-self.momentum)*batch_var
        )



    def normalize(
        self,
        tensor
    ):

        return torch.clamp(
            (
                tensor-self.mean
            )
            /
            (
                np.sqrt(self.var)+1e-8
            ),
            -5,
            5
        )

class LegionReplayBuffer:
    """
    Stores true (latent, action, reward, next_latent, done) transitions
    so the world model can be supervised correctly.
    """

    def __init__(
        self,
        max_size=100000
    ):

        self.buffer=deque(
            maxlen=max_size
        )


    def add(
        self,
        latent,
        action,
        reward,
        next_latent,
        done
    ):
        """
        latent, next_latent: torch tensors of shape (1, embed_dim), stored on CPU
        action: numpy array (action_dim,)
        reward: float
        done: float (0.0 or 1.0)
        """

        self.buffer.append(
            (
                latent.detach().cpu(),
                action,
                reward,
                next_latent.detach().cpu(),
                done
            )
        )


    def sample(
        self,
        batch_size
    ):

        actual_batch = min(batch_size, len(self.buffer))

        idx=np.random.choice(
            len(self.buffer),
            actual_batch,
            replace=False
        )

        return [
            self.buffer[i]
            for i in idx
        ]


    def __len__(self):

        return len(self.buffer)

class PixelPairBuffer:
    """
    Small ring buffer storing (current_pixels, action, next_pixels) pairs
    specifically for training the LeWM encoder via JEPA loss. Kept separate
    from LegionReplayBuffer (which stores latents) to avoid blowing up
    memory with raw image tensors at full buffer scale.
    """
    def __init__(self, max_size=2000):
        self.buffer = deque(maxlen=max_size)

    def add(self, current_pixels, action, next_pixels):
        # Store on CPU to avoid holding GPU memory for buffered frames
        self.buffer.append((
            current_pixels.detach().cpu(),
            action,
            next_pixels.detach().cpu(),
        ))

    def sample(self, batch_size):
        actual_batch = min(batch_size, len(self.buffer))
        idx = np.random.choice(len(self.buffer), actual_batch, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)
        
# ==============================================================
# Transition Helper
# ==============================================================


def process_legion_transition(
    env_info,
    raw_reward,
    reward_tracker
):
    """
    Processes reward signals before passing them
    into MuDreamer.
    """

    reward_tracker.update_statistics(
        [raw_reward]
    )


    clean_reward = reward_tracker.normalize(
        torch.tensor(
            [raw_reward],
            dtype=torch.float32
        )
    )


    is_success = bool(
        env_info.get(
            "success",
            0.0
        )
    )


    return clean_reward, is_success