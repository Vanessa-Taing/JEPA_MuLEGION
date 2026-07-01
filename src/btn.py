"""
Minimal Bilayer Tensor Network (BTN) for semantic triple decoding.

Based on: Tresp et al. "The Tensor Brain: A Unified Theory of Perception,
Memory and Semantic Decoding" (2023).

Architecture maps the MuDreamer latent state (the "global workspace" /
representation layer in BTN terminology) to semantic triples
(Subject, Predicate, Object) via soft attention over learned index-layer
embeddings. A GRU working memory accumulates triple history across the
imagination horizon and produces a conditioning vector for the actor.

This implements the SPM (Statement Prediction Model) path from Algorithm 1
of the paper in simplified form:
    latent -> subject_head -> subject_probs -> weighted subject embedding
    latent -> predicate_head -> ...
    latent -> object_head -> ...
    concat(s_embed, p_embed, o_embed) -> working_memory GRU -> actor conditioning
"""

import torch
import torch.nn as nn
import torch.optim as optim


# Fixed small vocabulary — sufficient for MetaWorld's structured domain.
# Subjects: robot parts / objects. Predicates: actions. Objects: locations / goals.
DEFAULT_VOCAB = {
    "subjects": [
        "gripper", "arm", "block", "door", "button", "handle",
        "end_effector", "target", "surface", "robot",
        "object_1", "object_2", "goal", "base", "wrist",
        "finger_left", "finger_right", "cube", "sphere", "peg",
        "slot", "drawer", "faucet", "window", "lever",
        "plate", "cup", "tool", "fixture", "environment"
    ],
    "predicates": [
        "reaches", "pushes", "grasps", "places", "opens",
        "closes", "presses", "lifts", "contacts", "approaches",
        "moves_toward", "moves_away", "aligns_with", "releases", "slides"
    ],
    "objects": [
        "target_position", "goal_region", "surface", "slot",
        "handle", "button", "door_frame", "block", "initial_position",
        "above", "below", "left_of", "right_of", "in_front_of",
        "gripper_open", "gripper_closed", "goal_achieved", "obstacle",
        "trajectory_waypoint", "contact_point", "rest_position",
        "pivot_point", "attachment_point", "end_effector_target",
        "workspace_boundary", "object_center", "grasp_point",
        "release_point", "approach_vector"
    ],
}


class MinimalBTN(nn.Module):
    """
    Simplified BTN: representation layer (latent) -> index layer (triple logits)
    -> working memory -> actor conditioning vector.

    The soft attention over embeddings is differentiable, so the triple
    decoding learns jointly with the actor via policy gradient without
    needing explicit triple supervision.
    """
    def __init__(
        self,
        embed_dim=512,
        n_subjects=30,
        n_predicates=15,
        n_objects=30,
        vocab_embed_dim=64,
        conditioning_dim=128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conditioning_dim = conditioning_dim
        self.n_subjects = n_subjects
        self.n_predicates = n_predicates
        self.n_objects = n_objects

        # Index layer: learnable concept embeddings (the BTN connection weights)
        self.subject_embeddings = nn.Embedding(n_subjects, vocab_embed_dim)
        self.predicate_embeddings = nn.Embedding(n_predicates, vocab_embed_dim)
        self.object_embeddings = nn.Embedding(n_objects, vocab_embed_dim)

        # Projection heads: representation layer -> index layer logits
        # These implement the SPM scoring function a_p^T g(a_o + g(a_s + g(a_t)))
        # in the simplified form of direct linear projections
        self.subject_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, n_subjects),
        )
        self.predicate_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, n_predicates),
        )
        self.object_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, n_objects),
        )

        # Working memory: accumulates episodic triple history across horizon
        # Analogous to the working memory layer h in the BTN paper (Section 5)
        # which enables binary relationship decoding requiring short-term storage
        self.working_memory = nn.GRUCell(
            vocab_embed_dim * 3, conditioning_dim
        )

        # Output projection to actor conditioning vector
        self.conditioning_proj = nn.Linear(conditioning_dim, conditioning_dim)

    def forward(self, latent, wm_hidden=None):
        """
        latent:    (B, embed_dim) -- the global workspace / representation layer
        wm_hidden: (B, conditioning_dim) or None

        Returns:
            conditioning: (B, conditioning_dim) -- semantic context for actor
            wm_hidden:    (B, conditioning_dim) -- updated working memory state
            triple_logits: dict for loss computation and logging
        """
        B = latent.shape[0]
        device = latent.device

        if wm_hidden is None:
            wm_hidden = torch.zeros(B, self.conditioning_dim, device=device)

        # Decode triple distributions from the representation layer (latent)
        subject_logits = self.subject_head(latent)      # (B, n_subjects)
        predicate_logits = self.predicate_head(latent)  # (B, n_predicates)
        object_logits = self.object_head(latent)        # (B, n_objects)

        # Soft attention over index layer embeddings — differentiable triple selection
        # This is the BTN oscillation: index layer activates representation layer
        # and vice versa, here implemented as a forward soft-lookup
        subject_probs = torch.softmax(subject_logits, dim=-1)
        predicate_probs = torch.softmax(predicate_logits, dim=-1)
        object_probs = torch.softmax(object_logits, dim=-1)

        subject_embed = subject_probs @ self.subject_embeddings.weight      # (B, D_v)
        predicate_embed = predicate_probs @ self.predicate_embeddings.weight
        object_embed = object_probs @ self.object_embeddings.weight

        # Concatenate to form the semantic triple embedding
        triple_embed = torch.cat(
            [subject_embed, predicate_embed, object_embed], dim=-1
        )  # (B, 3*D_v)

        # Update working memory with new triple
        wm_hidden = self.working_memory(triple_embed, wm_hidden)

        # Project accumulated triple context to actor conditioning vector
        conditioning = self.conditioning_proj(wm_hidden)

        return conditioning, wm_hidden, {
            "subject_logits": subject_logits,
            "predicate_logits": predicate_logits,
            "object_logits": object_logits,
        }

    @torch.no_grad()
    def decode_triple_hard(self, latent):
        """
        Hard argmax triple for logging only — not used during training.
        Returns indices into the vocabulary.
        """
        s = self.subject_head(latent).argmax(dim=-1)
        p = self.predicate_head(latent).argmax(dim=-1)
        o = self.object_head(latent).argmax(dim=-1)
        return s, p, o


class BTNTrainer:
    """
    Trains the BTN via two self-supervised objectives:

    1. Entropy regularization: prevents all frames collapsing to the same triple.
       High entropy is desired early — the BTN should spread probability mass
       across vocabulary before it has enough signal to commit.

    2. Consistency loss: similar latents should produce similar triples.
       Implemented as MSE between triple logits of a latent and the mean
       logits of its batch — pushes similar observations toward agreement.

    No explicit triple labels are needed. The BTN is supervised indirectly
    through the actor's policy gradient (conditioning improves actions ->
    actor loss decreases -> BTN weights update via the optimizer chain).
    """
    def __init__(
        self,
        btn,
        lr=3e-4,
        entropy_weight=0.1,
        consistency_weight=0.05,
        grad_clip=10.0,
    ):
        self.btn = btn
        self.optimizer = optim.Adam(self.btn.parameters(), lr=lr)
        self.entropy_weight = entropy_weight
        self.consistency_weight = consistency_weight
        self.grad_clip = grad_clip

    def train_step(self, latent_batch):
        """
        latent_batch: (B, embed_dim) -- batch of latent states from replay buffer
        """
        self.optimizer.zero_grad()

        # Forward pass — no wm_hidden for batch training (no sequence context)
        _, _, triple_logits = self.btn(latent_batch, wm_hidden=None)

        total_loss = torch.tensor(0.0, device=latent_batch.device)
        entropy_sum = torch.tensor(0.0, device=latent_batch.device)

        for key, logits in triple_logits.items():
            probs = torch.softmax(logits, dim=-1)
            # Entropy: H(p) = -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            entropy_sum = entropy_sum + entropy

            # Consistency: triple logits should be similar within the batch
            # (MetaWorld frames within the same task should share coarse structure)
            mean_logits = logits.mean(dim=0, keepdim=True).detach()
            consistency = torch.mean((logits - mean_logits) ** 2)
            total_loss = total_loss + self.consistency_weight * consistency

        # Maximize entropy (minimize negative entropy) — prevents collapse
        total_loss = total_loss - self.entropy_weight * entropy_sum

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.btn.parameters(), self.grad_clip)
        self.optimizer.step()

        return {"btn_loss": total_loss.item(), "btn_entropy": entropy_sum.item()}