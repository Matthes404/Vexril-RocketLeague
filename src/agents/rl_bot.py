"""
RL Bot agent for inference/testing
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pathlib import Path


class RLBot:
    """
    Rocket League bot that uses a trained PPO model for decision making
    """

    def __init__(self, model_path: str, vecnormalize_path: str = None, deterministic: bool = True):
        """
        Initialize the RL bot

        Args:
            model_path: Path to the trained model (.zip file)
            vecnormalize_path: Path to VecNormalize stats (.pkl file). If None, auto-detect.
            deterministic: If True, use deterministic actions (no exploration)
        """
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.model = None
        self.vec_normalize = None

        # Auto-detect VecNormalize path if not provided
        if vecnormalize_path is None:
            # Try to find corresponding vecnormalize file
            # e.g., models/rl_bot_final.zip -> models/rl_bot_vecnormalize.pkl
            model_dir = self.model_path.parent
            base_name = self.model_path.stem.replace('_final', '').replace('_latest', '').split('_')[0:2]
            base_name = '_'.join(base_name) if len(base_name) > 1 else base_name[0]
            vecnormalize_path = model_dir / f"{base_name}_vecnormalize.pkl"

        self.vecnormalize_path = Path(vecnormalize_path) if vecnormalize_path else None
        self.load_model()

    def load_model(self):
        """Load the trained model and VecNormalize stats if available"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"Loading model from {self.model_path}")
        self.model = PPO.load(str(self.model_path))
        print("Model loaded successfully")

        # Load VecNormalize stats if available
        if self.vecnormalize_path and self.vecnormalize_path.exists():
            print(f"Loading VecNormalize stats from {self.vecnormalize_path}")
            # Create a dummy environment for VecNormalize
            # We'll only use it for normalization, not actual env interaction
            from gymnasium import spaces

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
            print("VecNormalize stats loaded successfully")
        else:
            if self.vecnormalize_path:
                print(f"Warning: VecNormalize file not found at {self.vecnormalize_path}")
            print("Running without observation normalization")

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predict action given observation

        Args:
            observation: The current observation from the environment

        Returns:
            action: The predicted action
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Normalize observation if VecNormalize is available
        if self.vec_normalize is not None:
            # VecNormalize expects observations in shape (n_envs, obs_dim)
            obs_normalized = self.vec_normalize.normalize_obs(observation.reshape(1, -1))
            action, _ = self.model.predict(obs_normalized[0], deterministic=self.deterministic)
        else:
            action, _ = self.model.predict(observation, deterministic=self.deterministic)

        return action

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Alias for predict() for compatibility

        Args:
            obs: The current observation

        Returns:
            action: The predicted action
        """
        return self.predict(obs)


class RandomBot:
    """
    A simple random bot for testing/baseline comparison
    """

    def __init__(self):
        """Initialize random bot"""
        pass

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Return random actions

        Args:
            observation: The current observation (unused)

        Returns:
            action: Random action in the valid action space
        """
        # RLGym action space is typically 8-dimensional continuous
        # Each dimension is in [-1, 1]
        return np.random.uniform(-1, 1, size=8)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Alias for predict()

        Args:
            obs: The current observation

        Returns:
            action: Random action
        """
        return self.predict(obs)
