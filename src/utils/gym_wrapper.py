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

        # Get a sample observation to determine the space and number of agents
        sample_obs = self.env.reset()

        # Determine number of agents from observation structure
        # RLGym-Sim returns a list of observations when there are multiple agents
        if isinstance(sample_obs, list):
            self.num_agents = len(sample_obs)
            # Use first agent's observation as the sample
            sample_to_use = sample_obs[0]
        else:
            self.num_agents = 1
            sample_to_use = sample_obs

        # Flatten observation if needed
        if isinstance(sample_to_use, dict):
            # If obs is a dict, we need to flatten it
            sample_flat = self._flatten_obs(sample_to_use)
        elif isinstance(sample_to_use, (list, tuple)) and not isinstance(sample_obs, list):
            # If it's a list/tuple (but not multi-agent), convert to array
            sample_flat = np.array(sample_to_use).flatten()
        else:
            # Already an array
            sample_flat = np.array(sample_to_use).flatten()

        # Define observation space
        obs_dim = sample_flat.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Get action space from env and convert to gymnasium space
        if hasattr(self.env, 'action_space'):
            # Convert gym.spaces to gymnasium.spaces
            env_action_space = self.env.action_space
            # Create a new gymnasium Box space with the same parameters
            self.action_space = spaces.Box(
                low=env_action_space.low,
                high=env_action_space.high,
                shape=env_action_space.shape,
                dtype=env_action_space.dtype
            )
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

        # Handle multi-agent observations (return only first agent's obs)
        if isinstance(obs, list) and self.num_agents > 1:
            obs = obs[0]

        obs_flat = self._flatten_obs(obs).astype(np.float32)

        # Gymnasium API requires (obs, info) tuple
        return obs_flat, {}

    def step(self, action):
        """Step environment"""
        # Convert action to numpy array if needed
        action = np.array(action, dtype=np.float32)

        # Handle multi-agent: replicate action for all agents
        # The trained agent controls the first agent, others get zero actions
        if self.num_agents > 1:
            # Create actions for all agents
            actions = [action]
            # Add zero actions for opponent agents
            for _ in range(self.num_agents - 1):
                actions.append(np.zeros_like(action))
            action = np.array(actions)

        # RLGym returns (obs, reward, done, info)
        obs, reward, done, info = self.env.step(action)

        # Handle multi-agent observations (extract first agent's obs)
        if isinstance(obs, list) and self.num_agents > 1:
            obs = obs[0]

        # Handle multi-agent rewards (extract first agent's reward)
        if isinstance(reward, (list, np.ndarray)) and self.num_agents > 1:
            reward = reward[0]

        # Handle multi-agent dones (extract first agent's done)
        if isinstance(done, (list, np.ndarray)) and self.num_agents > 1:
            done = done[0]

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
