"""
Enhanced Observation Builder for RLGym-Sim.

This observation builder includes boost pad states and game context,
replacing the zero-padding in DefaultObs with useful information.

Observation Format (89 dims):
- Ball: 9 dims (position, velocity, angular velocity)
- Self: 20 dims (position, velocity, forward, up, angular vel, boost, flags)
- Boost Pads: 34 dims (active status for all 34 pads)
- Game State: 6 dims (goal diff, time left, overtime, demo self, demo opp, padding)
- Opponent: 20 dims (same as self)
"""
import numpy as np
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.gamestates import PlayerData, GameState
import math


class EnhancedObs(ObsBuilder):
    """
    Enhanced observation builder that includes:
    - Ball state
    - Player state (self)
    - All 34 boost pad states (active/inactive)
    - Game context (score, time, overtime, demo states)
    - Opponent state

    Total: 89 dimensions (same as DefaultObs padded format)
    """

    # Normalization constants
    POS_STD = 2300.0  # Arena half-length approximately
    VEL_STD = 2300.0  # Max car speed approximately
    ANG_STD = math.pi  # Max angular value

    # Boost pad locations (all 34 pads in RLBot order)
    # Format: (x, y, z, is_big)
    BOOST_PADS = [
        # Blue side back
        (0.0, -4240.0, 70.0, False),
        (-1792.0, -4184.0, 70.0, False),
        (1792.0, -4184.0, 70.0, False),
        (-3072.0, -4096.0, 73.0, True),   # Big boost
        (3072.0, -4096.0, 73.0, True),    # Big boost
        # Blue side mid-back
        (-940.0, -3308.0, 70.0, False),
        (940.0, -3308.0, 70.0, False),
        (0.0, -2816.0, 70.0, False),
        (-3584.0, -2484.0, 70.0, False),
        (3584.0, -2484.0, 70.0, False),
        # Blue side mid
        (-1788.0, -2300.0, 70.0, False),
        (1788.0, -2300.0, 70.0, False),
        (-2048.0, -1036.0, 70.0, False),
        (0.0, -1024.0, 70.0, False),
        (2048.0, -1036.0, 70.0, False),
        # Center
        (-3584.0, 0.0, 73.0, True),       # Big boost
        (-1024.0, 0.0, 70.0, False),
        (1024.0, 0.0, 70.0, False),
        (3584.0, 0.0, 73.0, True),        # Big boost
        # Orange side mid
        (-2048.0, 1036.0, 70.0, False),
        (0.0, 1024.0, 70.0, False),
        (2048.0, 1036.0, 70.0, False),
        (-1788.0, 2300.0, 70.0, False),
        (1788.0, 2300.0, 70.0, False),
        # Orange side mid-back
        (-3584.0, 2484.0, 70.0, False),
        (3584.0, 2484.0, 70.0, False),
        (0.0, 2816.0, 70.0, False),
        (-940.0, 3308.0, 70.0, False),
        (940.0, 3308.0, 70.0, False),
        # Orange side back
        (-3072.0, 4096.0, 73.0, True),    # Big boost
        (3072.0, 4096.0, 73.0, True),     # Big boost
        (-1792.0, 4184.0, 70.0, False),
        (1792.0, 4184.0, 70.0, False),
        (0.0, 4240.0, 70.0, False),
    ]

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        """Called at the start of each episode"""
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """
        Build observation for a single player.

        Args:
            player: The player to build observation for
            state: Current game state
            previous_action: Previous action taken (unused in this builder)

        Returns:
            89-dimensional observation array
        """
        obs = []

        # Team-based inversion (orange team sees inverted coordinates)
        inv = -1 if player.team_num == 1 else 1

        # === BALL STATE (9 dims) ===
        ball = state.ball
        obs.extend([
            ball.position[0] / self.POS_STD * inv,
            ball.position[1] / self.POS_STD * inv,
            ball.position[2] / self.POS_STD,
            ball.linear_velocity[0] / self.VEL_STD * inv,
            ball.linear_velocity[1] / self.VEL_STD * inv,
            ball.linear_velocity[2] / self.VEL_STD,
            ball.angular_velocity[0] / self.ANG_STD,
            ball.angular_velocity[1] / self.ANG_STD,
            ball.angular_velocity[2] / self.ANG_STD,
        ])

        # === SELF STATE (20 dims) ===
        obs.extend(self._get_player_obs(player, inv))

        # === BOOST PADS (34 dims) ===
        # Get boost pad states from game state
        for i, pad_info in enumerate(self.BOOST_PADS):
            if i < len(state.boost_pads):
                # 1.0 if active (can be picked up), 0.0 if inactive
                obs.append(1.0 if state.boost_pads[i] else 0.0)
            else:
                # If we don't have boost pad info, assume active
                obs.append(1.0)

        # === GAME STATE (6 dims) ===
        # Goal difference (from this player's perspective)
        if player.team_num == 0:  # Blue team
            goal_diff = state.blue_score - state.orange_score
        else:  # Orange team
            goal_diff = state.orange_score - state.blue_score
        obs.append(np.clip(goal_diff / 5.0, -1.0, 1.0))

        # Time remaining (normalized, 0-1)
        # RLGym-Sim doesn't always provide game time, default to 0.5 if unavailable
        if hasattr(state, 'game_timer') and state.game_timer is not None:
            time_left = np.clip(state.game_timer / 300.0, 0.0, 1.0)
        else:
            time_left = 0.5
        obs.append(time_left)

        # Is overtime
        is_overtime = 1.0 if hasattr(state, 'is_overtime') and state.is_overtime else 0.0
        obs.append(is_overtime)

        # Demo state for self (1.0 if demolished, 0.0 otherwise)
        demo_self = 1.0 if player.is_demoed else 0.0
        obs.append(demo_self)

        # Demo state for opponent (first opponent)
        demo_opp = 0.0
        for p in state.players:
            if p.team_num != player.team_num:
                demo_opp = 1.0 if p.is_demoed else 0.0
                break
        obs.append(demo_opp)

        # Padding (1 dim) - could be used for future features
        obs.append(0.0)

        # === OPPONENT STATE (20 dims) ===
        opponent = None
        for p in state.players:
            if p.team_num != player.team_num:
                opponent = p
                break

        if opponent is not None:
            obs.extend(self._get_player_obs(opponent, inv))
        else:
            # Pad with zeros if no opponent
            obs.extend([0.0] * 20)

        return np.array(obs, dtype=np.float32)

    def _get_player_obs(self, player: PlayerData, inv: int) -> list:
        """
        Get observation data for a player.

        Returns 20 dimensions:
        - Position (3)
        - Velocity (3)
        - Forward vector (3)
        - Up vector (3)
        - Angular velocity (3)
        - Boost (1)
        - Flags: on_ground, is_supersonic, jumped, double_jumped (4)
        """
        obs = []

        # Position
        obs.extend([
            player.car_data.position[0] / self.POS_STD * inv,
            player.car_data.position[1] / self.POS_STD * inv,
            player.car_data.position[2] / self.POS_STD,
        ])

        # Velocity
        obs.extend([
            player.car_data.linear_velocity[0] / self.VEL_STD * inv,
            player.car_data.linear_velocity[1] / self.VEL_STD * inv,
            player.car_data.linear_velocity[2] / self.VEL_STD,
        ])

        # Forward vector (from rotation matrix)
        forward = player.car_data.forward()
        obs.extend([
            forward[0] * inv,
            forward[1] * inv,
            forward[2],
        ])

        # Up vector (from rotation matrix)
        up = player.car_data.up()
        obs.extend([
            up[0] * inv,
            up[1] * inv,
            up[2],
        ])

        # Angular velocity
        obs.extend([
            player.car_data.angular_velocity[0] / self.ANG_STD,
            player.car_data.angular_velocity[1] / self.ANG_STD,
            player.car_data.angular_velocity[2] / self.ANG_STD,
        ])

        # Boost (0-1)
        obs.append(player.boost_amount)

        # Flags
        obs.append(1.0 if player.on_ground else 0.0)
        # Calculate supersonic from velocity (threshold ~2200 uu/s)
        speed = np.linalg.norm(player.car_data.linear_velocity)
        obs.append(1.0 if speed >= 2200.0 else 0.0)
        obs.append(1.0 if player.has_jump else 0.0)
        obs.append(1.0 if player.has_flip else 0.0)

        return obs
