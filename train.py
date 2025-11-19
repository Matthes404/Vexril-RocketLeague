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
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

from src.rewards.custom_reward import CustomReward
from src.state_setters.custom_state_setter import CustomStateSetter
from src.utils import RLGymSimWrapper


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
    env = RLGymSimWrapper(rlgym_env)

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=str(models_dir),
        name_prefix=config['training']['model_name']
    )

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
            device=device,
            verbose=1,
            tensorboard_log=str(logs_dir)
        )

    # Train the model
    print(f"Starting training for {config['training']['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save final model
    final_model_path = models_dir / f"{config['training']['model_name']}_final.zip"
    model.save(str(final_model_path))
    print(f"Training complete! Final model saved to {final_model_path}")

    env.close()


if __name__ == "__main__":
    main()
