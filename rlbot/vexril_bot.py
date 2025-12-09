"""
Vexril Bot - RLBot integration for PPO-trained Rocket League bot.

This bot loads a model trained with RLGym-Sim and Stable-Baselines3 PPO,
and uses it to play in actual Rocket League via RLBotGUI.
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import from src/
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
from gymnasium import spaces

from state_converter import StateConverter, model_action_to_controller


class VexrilBot(BaseAgent):
    """
    RLBot agent that uses a PPO model trained with RLGym-Sim.
    """

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.model = None
        self.vec_normalize = None
        self.state_converter = StateConverter()
        self.model_loaded = False
        self.model_path = None
        self.vecnormalize_path = None
        self.first_tick = True  # For diagnostic logging
        self.tick_count = 0

    def initialize_agent(self):
        """
        Called when the bot is first initialized.
        Load the trained PPO model here.
        """
        # Set team for state converter
        self.state_converter.set_team(self.team)

        # Try to find the model
        # Priority order:
        # 1. models/rl_bot_final.zip (final trained model)
        # 2. models/rl_bot_latest.zip (latest checkpoint)
        # 3. Any checkpoint in models/ directory

        models_dir = Path(__file__).parent.parent / 'models'

        # Check for final model
        final_model = models_dir / 'rl_bot_final.zip'
        latest_model = models_dir / 'rl_bot_latest.zip'

        if final_model.exists():
            self.model_path = final_model
        elif latest_model.exists():
            self.model_path = latest_model
        else:
            # Look for any checkpoint
            checkpoints = sorted(models_dir.glob('rl_bot_*.zip'))
            if checkpoints:
                # Use the most recent checkpoint (highest step count)
                self.model_path = checkpoints[-1]
            else:
                self.logger.error("No trained model found in models/ directory!")
                self.logger.error("Please train a model first using train.py")
                return

        # Load the model
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            self.model = PPO.load(str(self.model_path))
            self.model_loaded = True
            self.logger.info("Model loaded successfully!")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            return

        # Load VecNormalize stats (CRITICAL for correct predictions)
        # The model was trained with normalized observations, so we must normalize them here too
        self.vecnormalize_path = models_dir / 'rl_bot_vecnormalize.pkl'

        if self.vecnormalize_path.exists():
            try:
                self.logger.info(f"Loading VecNormalize stats from: {self.vecnormalize_path}")

                # Create a minimal dummy environment for VecNormalize
                class DummyEnvForNorm:
                    """Minimal env class for VecNormalize loading"""
                    def __init__(self, obs_space):
                        self.observation_space = obs_space
                        self.action_space = spaces.Box(low=-1, high=1, shape=(8,))

                    def reset(self):
                        return np.zeros(self.observation_space.shape)

                    def step(self, action):
                        return np.zeros(self.observation_space.shape), 0.0, False, False, {}

                # Get observation space from model
                obs_dim = self.model.observation_space.shape[0]
                dummy_env = DummyEnvForNorm(spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)))
                wrapped_env = DummyVecEnv([lambda: dummy_env])

                self.vec_normalize = VecNormalize.load(str(self.vecnormalize_path), wrapped_env)
                # Don't update stats during inference
                self.vec_normalize.training = False
                self.vec_normalize.norm_reward = False

                self.logger.info("VecNormalize stats loaded successfully!")
            except Exception as e:
                self.logger.warning(f"Failed to load VecNormalize stats: {e}")
                self.logger.warning("Running without observation normalization (bot may perform poorly)")
        else:
            self.logger.warning(f"VecNormalize file not found at: {self.vecnormalize_path}")
            self.logger.warning("Running without observation normalization (bot may perform poorly)")

        # Log what match type this model expects
        expected_obs_size = self.model.observation_space.shape[0]
        match_types = {
            29: "1v0 (solo practice)",
            49: "1v1",
            69: "1v2",
            89: "2v2",
            109: "2v3 or 3v2",
            129: "3v3"
        }
        expected_type = match_types.get(expected_obs_size, f"unknown ({expected_obs_size} dims)")
        self.logger.info("=" * 60)
        self.logger.info(f"MODEL TRAINED FOR: {expected_type}")
        self.logger.info(f"PLEASE SET UP A {expected_type.upper()} MATCH IN RLBOTGUI!")
        self.logger.info("=" * 60)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        Called every game tick to get the bot's controller input.

        Args:
            packet: Current game state from RLBot

        Returns:
            Controller state for this bot
        """
        # If model not loaded, return no input
        if not self.model_loaded or self.model is None:
            return SimpleControllerState()

        try:
            # Convert game state to observation format
            observation = self.state_converter.packet_to_observation(packet, self.index)
            self.tick_count += 1

            # Check if observation size matches model expectations
            expected_obs_size = self.model.observation_space.shape[0]
            actual_obs_size = observation.shape[0]

            # Log diagnostic info on first tick
            if self.first_tick:
                self.first_tick = False
                self.logger.info("=" * 60)
                self.logger.info("FIRST TICK DIAGNOSTICS")
                self.logger.info(f"Observation size: {actual_obs_size} (expected: {expected_obs_size})")
                self.logger.info(f"Observation stats:")
                self.logger.info(f"  min: {observation.min():.4f}, max: {observation.max():.4f}")
                self.logger.info(f"  mean: {observation.mean():.4f}, std: {observation.std():.4f}")
                self.logger.info(f"  first 10 values: {observation[:10]}")
                self.logger.info(f"VecNormalize loaded: {self.vec_normalize is not None}")
                if self.vec_normalize is not None:
                    self.logger.info(f"  obs_rms mean shape: {self.vec_normalize.obs_rms.mean.shape}")
                self.logger.info("=" * 60)

            if actual_obs_size != expected_obs_size:
                # Calculate what match type this is
                match_types = {
                    29: "1v0 (solo)",
                    49: "1v1",
                    69: "1v2",
                    89: "2v2",
                    109: "2v3 or 3v2",
                    129: "3v3"
                }
                actual_type = match_types.get(actual_obs_size, f"unknown ({actual_obs_size} dims)")
                expected_type = match_types.get(expected_obs_size, f"unknown ({expected_obs_size} dims)")

                self.logger.error("=" * 60)
                self.logger.error("OBSERVATION SIZE MISMATCH!")
                self.logger.error(f"Model expects: {expected_obs_size} dimensions ({expected_type})")
                self.logger.error(f"Current match: {actual_obs_size} dimensions ({actual_type})")
                self.logger.error(f"Please set up a {expected_type.upper()} match in RLBotGUI!")
                self.logger.error("=" * 60)
                return SimpleControllerState()

            # Normalize observation if VecNormalize is available
            # CRITICAL: The model was trained on normalized observations!
            if self.vec_normalize is not None:
                # VecNormalize expects observations in shape (n_envs, obs_dim)
                obs_normalized = self.vec_normalize.normalize_obs(observation.reshape(1, -1))
                action, _ = self.model.predict(obs_normalized[0], deterministic=True)
            else:
                # No normalization (will perform poorly if model was trained with VecNormalize)
                action, _ = self.model.predict(observation, deterministic=True)

            # Convert model action to controller state
            controller = model_action_to_controller(action)

            # Update previous action for next observation (some obs builders use this)
            self.state_converter.update_previous_action(action)

            return controller

        except Exception as e:
            # Log error and return safe default
            self.logger.error(f"Error getting model output: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return SimpleControllerState()

    def retire(self):
        """Called when the bot is being shut down"""
        self.logger.info("Vexril Bot shutting down...")


# For backward compatibility with older RLBot versions
class Agent(VexrilBot):
    """Alias for VexrilBot to support different naming conventions"""
    pass
