from torch.utils.tensorboard import SummaryWriter


class JepaDreamerLogger:
    """
    Handles specialized monitoring loops for un-decoded model training,
    exposing representation collapses or policy instability instantly.
    """
    def __init__(self, log_dir="./logs/legion_jepa_dreamer"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_imagination_step(self, step, actor_loss, critic_loss, pred_loss, policy_entropy, rewards):
        """Logs standard training statistics to evaluate the current skill core."""
        self.writer.add_scalar("MuDreamer/Loss/Actor", actor_loss, step)
        self.writer.add_scalar("MuDreamer/Loss/Critic", critic_loss, step)
        self.writer.add_scalar("LeWM/Loss/Predictor_Drift", pred_loss, step)
        self.writer.add_scalar("MuDreamer/Policy_Entropy", policy_entropy, step)
        self.writer.add_scalar("Environment/Normalized_Reward_Mean", rewards.mean().item(), step)

    def log_task_expansion(self, step, active_task_id, new_total_components, similarity_score):
        """Records structural growth operations triggered by the LEGION framework."""
        self.writer.add_scalar("LEGION/Task_Component_Count", new_total_components, step)
        self.writer.add_scalar(f"LEGION/Similarity/Task_{active_task_id}", similarity_score, step)

    def log_cluster_diagnostics(self, step, num_components, flat_cluster_sizes, seq_cluster_episode_counts=None):
        """
        Periodic (not just birth-triggered) snapshot of DPMM cluster state,
        covering BOTH replay mechanisms used in the RSSM/BTN pipeline:
          - flat_cluster_sizes: {task_id: transition_count} from LegionReplayBuffer
            (imagination-seeding data)
          - seq_cluster_episode_counts: {task_id: episode_count} from
            SequenceReplayBuffer (RSSM sequence-training data)

        Logged independently of births so cluster count is visible over time
        even on runs where zero births happen after warmup. On a single-task
        run (e.g. training only on reach-v3), num_components should stay at 1
        — if it climbs above 1, the DPMM is fragmenting one task into several
        undertrained heads, which is the diagnostic signal for the
        encoder-drift-causes-spurious-births hypothesis.
        """
        self.writer.add_scalar("LEGION/Diagnostic/Num_Active_Clusters", num_components, step)

        if flat_cluster_sizes:
            self.writer.add_scalar(
                "LEGION/Diagnostic/Flat_Largest_Cluster_Size", max(flat_cluster_sizes.values()), step
            )
            self.writer.add_scalar(
                "LEGION/Diagnostic/Flat_Smallest_Cluster_Size", min(flat_cluster_sizes.values()), step
            )
            for tid, size in flat_cluster_sizes.items():
                self.writer.add_scalar(f"LEGION/Diagnostic/FlatClusterSize_{tid}", size, step)

        if seq_cluster_episode_counts:
            self.writer.add_scalar(
                "LEGION/Diagnostic/Seq_Largest_Cluster_Episodes",
                max(seq_cluster_episode_counts.values()),
                step,
            )
            self.writer.add_scalar(
                "LEGION/Diagnostic/Seq_Smallest_Cluster_Episodes",
                min(seq_cluster_episode_counts.values()),
                step,
            )
            for tid, count in seq_cluster_episode_counts.items():
                self.writer.add_scalar(f"LEGION/Diagnostic/SeqClusterEpisodes_{tid}", count, step)

    def log_eval_precision(self, step, mean_final_distance):
        """
        Mean distance-to-goal at episode end during eval, separate from
        reward/success. Distinguishes 'policy behaves coarsely correctly but
        can't hit the success radius' (this metric shrinks but stays above
        threshold) from 'policy isn't heading toward the goal at all' (this
        metric stays flat/high).
        """
        self.writer.add_scalar("Eval/Diagnostic/MeanFinalDistanceToGoal", mean_final_distance, step)

    def close(self):
        self.writer.close()