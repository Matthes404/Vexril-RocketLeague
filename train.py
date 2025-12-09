"""
Main training script for the Rocket League RL Bot using RLGym-Sim
"""
import os
import yaml
from pathlib import Path
import rlgym_sim
from rlgym_sim.utils.gamestates import GameState
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.obs_builders import DefaultObs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from src.rewards.custom_reward import CustomReward
from src.state_setters.custom_state_setter import CustomStateSetter
from src.utils import RLGymSimWrapper


class VecNormalizeCallback(BaseCallback):
    """
    Callback to save VecNormalize stats periodically
    """
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(self.save_path)
                if self.verbose > 0:
                    print(f"Saved VecNormalize stats at step {self.num_timesteps}")
        return True


def load_config(config_path: str = "configs/training_config.yaml"):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_rlgym_env(config):
    """Create and configure the RLGym-Sim environment"""
    env = rlgym_sim.make(
        tick_skip=config['env']['tick_skip'],
        spawn_opponents=config['env']['spawn_opponents'],
        team_size=config['env']['team_size'],
        terminal_conditions=[
            TimeoutCondition(config['env']['timeout_seconds']),
            GoalScoredCondition()
        ],
        reward_fn=CustomReward(),
        obs_builder=DefaultObs(),  # Default observation builder
        state_setter=CustomStateSetter(),
        copy_gamestate_every_step=True  # Improves performance
    )

    return env


def main():
    """Main training loop"""
    # Load configuration
    config = load_config()

    # Check for GPU availability
    # Note: PPO with MlpPolicy works better on CPU for this use case
    # GPU is mainly beneficial for CNN policies
    device = "cpu"  # Force CPU for better performance with MlpPolicy
    print(f"Using device: {device}")

    # Create directories
    models_dir = Path(config['paths']['models_dir'])
    logs_dir = Path(config['paths']['logs_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    print("Creating RLGym environment...")
    rlgym_env = create_rlgym_env(config)

    # Wrap in Gymnasium-compatible wrapper for SB3
    # Use configured opponent policy (default: "mixed" for varied opposition)
    opponent_policy = config['env'].get('opponent_policy', 'mixed')
    print(f"Using opponent policy: {opponent_policy}")
    base_env = RLGymSimWrapper(rlgym_env, opponent_policy=opponent_policy)

    # Wrap in DummyVecEnv (required for VecNormalize)
    env = DummyVecEnv([lambda: base_env])

    # CRITICAL: Add VecNormalize for observation and reward normalization
    # This will dramatically improve learning stability and value function performance
    vecnorm_path = models_dir / f"{config['training']['model_name']}_vecnormalize.pkl"

    if vecnorm_path.exists() and config['training']['resume_training']:
        print(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(str(vecnorm_path), env)
    else:
        print("Creating new VecNormalize wrapper...")
        env = VecNormalize(
            env,
            norm_obs=True,  # Normalize observations
            norm_reward=True,  # Normalize rewards
            clip_obs=10.0,  # Clip normalized observations to [-10, 10]
            clip_reward=10.0,  # Clip normalized rewards to [-10, 10]
            gamma=config['ppo']['gamma'],  # Use same gamma as PPO
            epsilon=1e-8
        )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=str(models_dir),
        name_prefix=config['training']['model_name']
    )

    vecnormalize_callback = VecNormalizeCallback(
        save_freq=config['training']['save_freq'],
        save_path=str(vecnorm_path),
        verbose=1
    )

    callbacks = [checkpoint_callback, vecnormalize_callback]

    # Initialize or load model
    model_path = models_dir / f"{config['training']['model_name']}_latest.zip"

    if model_path.exists() and config['training']['resume_training']:
        print(f"Loading existing model from {model_path}")
        model = PPO.load(
            str(model_path),
            env=env,
            device=device,
            tensorboard_log=str(logs_dir)
        )
    else:
        print("Creating new PPO model...")
        # Define policy network architecture
        # Larger networks can learn more complex behaviors
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256, 256],  # Policy network: 3 layers of 256 units
                vf=[256, 256, 256]   # Value network: 3 layers of 256 units
            ),
            activation_fn=torch.nn.ReLU,  # ReLU activation
            # Initialize with orthogonal initialization (better for RL)
            ortho_init=True,
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config['ppo']['learning_rate'],
            n_steps=config['ppo']['n_steps'],
            batch_size=config['ppo']['batch_size'],
            n_epochs=config['ppo']['n_epochs'],
            gamma=config['ppo']['gamma'],
            gae_lambda=config['ppo']['gae_lambda'],
            clip_range=config['ppo']['clip_range'],
            ent_coef=config['ppo']['ent_coef'],
            vf_coef=config['ppo']['vf_coef'],
            max_grad_norm=config['ppo']['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
            tensorboard_log=str(logs_dir)
        )

    # Train the model
    print(f"Starting training for {config['training']['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callbacks,
        progress_bar=True
    )

    # Save final model
    final_model_path = models_dir / f"{config['training']['model_name']}_final.zip"
    model.save(str(final_model_path))
    print(f"Training complete! Final model saved to {final_model_path}")

    # Save VecNormalize stats
    env.save(str(vecnorm_path))
    print(f"VecNormalize stats saved to {vecnorm_path}")

    env.close()


if __name__ == "__main__":
    main()
