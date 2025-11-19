"""
RL Bot agent for inference/testing
"""
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path


class RLBot:
    """
    Rocket League bot that uses a trained PPO model for decision making
    """

    def __init__(self, model_path: str, deterministic: bool = True):
        """
        Initialize the RL bot

        Args:
            model_path: Path to the trained model (.zip file)
            deterministic: If True, use deterministic actions (no exploration)
        """
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        print(f"Loading model from {self.model_path}")
        self.model = PPO.load(str(self.model_path))
        print("Model loaded successfully")

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
