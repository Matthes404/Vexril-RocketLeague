"""
Gymnasium wrapper for RLGym-Sim environment compatibility with Stable-Baselines3
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RLGymSimWrapper(gym.Env):
    """
    Wrapper to make RLGym-Sim environment compatible with SB3
    Handles observation flattening and Gymnasium API compatibility
    """

    def __init__(self, rlgym_env):
        """
        Initialize wrapper

        Args:
            rlgym_env: RLGym-Sim environment instance
        """
        super().__init__()
        self.env = rlgym_env

        # Get a sample observation to determine the space
        sample_obs = self.env.reset()

        # Flatten observation if needed
        if isinstance(sample_obs, dict):
            # If obs is a dict, we need to flatten it
            sample_flat = self._flatten_obs(sample_obs)
        elif isinstance(sample_obs, (list, tuple)):
            # If it's a list/tuple, convert to array
            sample_flat = np.array(sample_obs).flatten()
        else:
            # Already an array
            sample_flat = np.array(sample_obs).flatten()

        # Define observation space
        obs_dim = sample_flat.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Get action space from env
        if hasattr(self.env, 'action_space'):
            self.action_space = self.env.action_space
        else:
            # Default action space for RLGym (8-dimensional continuous)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(8,),
                dtype=np.float32
            )

    def _flatten_obs(self, obs):
        """Flatten observation to 1D array"""
        if isinstance(obs, dict):
            # Concatenate all dict values
            return np.concatenate([np.array(v).flatten() for v in obs.values()])
        elif isinstance(obs, (list, tuple)):
            return np.array(obs).flatten()
        else:
            return np.array(obs).flatten()

    def reset(self, seed=None, options=None):
        """Reset environment"""
        # RLGym-Sim reset doesn't take seed parameter
        obs = self.env.reset()
        obs_flat = self._flatten_obs(obs).astype(np.float32)

        # Gymnasium API requires (obs, info) tuple
        return obs_flat, {}

    def step(self, action):
        """Step environment"""
        # Convert action to numpy array if needed
        action = np.array(action, dtype=np.float32)

        # RLGym returns (obs, reward, done, info)
        obs, reward, done, info = self.env.step(action)

        # Flatten observation
        obs_flat = self._flatten_obs(obs).astype(np.float32)

        # Gymnasium API requires (obs, reward, terminated, truncated, info)
        # For compatibility, we use done for both terminated and truncated
        terminated = done
        truncated = False

        return obs_flat, float(reward), terminated, truncated, info

    def render(self):
        """Render environment"""
        if hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        """Close environment"""
        if hasattr(self.env, 'close'):
            self.env.close()
