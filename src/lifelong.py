import numpy as np
import torch
from collections import deque

class LegionComponentAllocator:
    """
    Improved LEGION allocator.

    Uses:
    - normalized latent embeddings
    - exponential moving average centroid
    - novelty detection
    - stable task assignment
    """

    def __init__(
        self,
        threshold=0.80,
        ema_alpha=0.05,
        max_components=50,
        warmup_steps=50
    ):

        self.expansion_threshold = threshold
        self.ema_alpha = ema_alpha
        self.max_components = max_components
        self.warmup_steps = warmup_steps

        self.component_profiles = {}



    def normalize_latent(self, latent):

        norm = np.linalg.norm(latent)

        if norm < 1e-8:
            return latent

        return latent / norm



    def compute_cosine_similarity(
        self,
        a,
        b
    ):

        return np.dot(a,b)



    def update_centroid(
        self,
        centroid,
        new_latent
    ):

        updated = (
            (1-self.ema_alpha)*centroid
            +
            self.ema_alpha*new_latent
        )

        return self.normalize_latent(updated)



    def evaluate_task_assignment(
        self,
        current_latent_tensor,
        step_idx,
        logger
    ):


        latent_vector = (
            current_latent_tensor
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )


        latent_vector = self.normalize_latent(
            latent_vector
        )


        #
        # First task
        #
        if len(self.component_profiles)==0:


            self.component_profiles[0]={
                "centroid":latent_vector.copy(),
                "count":1
            }


            logger.log_task_expansion(
                step_idx,
                active_task_id=0,
                new_total_components=1,
                similarity_score=1.0
            )


            return 0,True



        #
        # Search closest component
        #

        best_score=-1
        best_id=None


        for cid,profile in self.component_profiles.items():

            score=self.compute_cosine_similarity(
                latent_vector,
                profile["centroid"]
            )

            if score>best_score:
                best_score=score
                best_id=cid



        #
        # Existing knowledge
        #

        if (
            best_score >= self.expansion_threshold
            or step_idx < self.warmup_steps
        ):


            profile=self.component_profiles[best_id]


            profile["centroid"]=self.update_centroid(
                profile["centroid"],
                latent_vector
            )


            profile["count"]+=1


            return best_id,False



        #
        # New knowledge
        #

        if len(self.component_profiles)>=self.max_components:

            return best_id,False



        new_id=len(self.component_profiles)


        self.component_profiles[new_id]={

            "centroid":latent_vector.copy(),

            "count":1

        }



        logger.log_task_expansion(
            step_idx,
            active_task_id=new_id,
            new_total_components=new_id+1,
            similarity_score=best_score
        )


        return new_id,True


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


    def __init__(
        self,
        max_size=100000
    ):

        self.buffer=deque(
            maxlen=max_size
        )


    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done
    ):

        self.buffer.append(
            (
                obs,
                action,
                reward,
                next_obs,
                done
            )
        )


    def sample(
        self,
        batch_size
    ):

        idx=np.random.choice(
            len(self.buffer),
            batch_size
        )

        return [
            self.buffer[i]
            for i in idx
        ]


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