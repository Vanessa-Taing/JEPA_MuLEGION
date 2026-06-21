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
        
    def close(self):
        self.writer.close()
