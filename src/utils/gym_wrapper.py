"""
Gymnasium wrapper for RLGym-Sim environment compatibility with Stable-Baselines3
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OpponentPolicy:
    """
    Base class for opponent policies during training.
    Subclass this to create different opponent behaviors.
    """

    def get_action(self, obs) -> np.ndarray:
        """Return an 8D action array for the opponent"""
        raise NotImplementedError


class ZeroOpponent(OpponentPolicy):
    """Opponent that does nothing (original behavior - NOT recommended)"""

    def get_action(self, obs) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)


class RandomOpponent(OpponentPolicy):
    """Opponent that takes random actions (basic exploration)"""

    def get_action(self, obs) -> np.ndarray:
        return np.random.uniform(-1, 1, size=8).astype(np.float32)


class BasicChaseOpponent(OpponentPolicy):
    """
    Simple opponent that tries to drive towards the ball.
    This provides basic opposition without being too sophisticated.
    """

    def get_action(self, obs) -> np.ndarray:
        # Create action array
        action = np.zeros(8, dtype=np.float32)

        # Always drive forward
        action[0] = 1.0  # throttle

        # Add some random steering for variety
        action[1] = np.random.uniform(-0.3, 0.3)

        # Occasionally boost
        if np.random.random() < 0.3:
            action[6] = 1.0  # boost

        # Occasionally jump (for aerial attempts)
        if np.random.random() < 0.05:
            action[5] = 1.0  # jump

        return action


class MixedOpponent(OpponentPolicy):
    """
    Opponent that switches between different behaviors.
    Provides varied opposition during training.
    """

    def __init__(self):
        self.policies = [RandomOpponent(), BasicChaseOpponent(), ZeroOpponent()]
        self.current_policy = 0
        self.steps = 0
        self.switch_interval = 500  # Switch policy every N steps

    def get_action(self, obs) -> np.ndarray:
        self.steps += 1
        if self.steps >= self.switch_interval:
            self.steps = 0
            self.current_policy = np.random.randint(0, len(self.policies))
        return self.policies[self.current_policy].get_action(obs)


class RLGymSimWrapper(gym.Env):
    """
    Wrapper to make RLGym-Sim environment compatible with SB3
    Handles observation flattening and Gymnasium API compatibility
    """

    def __init__(self, rlgym_env, opponent_policy: str = "mixed"):
        """
        Initialize wrapper

        Args:
            rlgym_env: RLGym-Sim environment instance
            opponent_policy: Type of opponent policy to use during training.
                Options: "zero" (frozen), "random", "chase", "mixed" (default)
                IMPORTANT: "zero" is NOT recommended - opponents won't move!
        """
        super().__init__()
        self.env = rlgym_env

        # Initialize opponent policy
        opponent_policies = {
            "zero": ZeroOpponent,
            "random": RandomOpponent,
            "chase": BasicChaseOpponent,
            "mixed": MixedOpponent
        }
        if opponent_policy not in opponent_policies:
            print(f"WARNING: Unknown opponent policy '{opponent_policy}', using 'mixed'")
            opponent_policy = "mixed"
        self.opponent = opponent_policies[opponent_policy]()
        print(f"Training opponent policy: {opponent_policy}")

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

        # Handle multi-agent: trained agent controls first agent,
        # opponents use the configured opponent policy
        if self.num_agents > 1:
            # Create actions for all agents
            actions = [action]
            # Add opponent actions (NOT zero - that's the old bug!)
            for _ in range(self.num_agents - 1):
                opponent_action = self.opponent.get_action(None)
                actions.append(opponent_action)
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
