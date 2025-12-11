"""
State converter for RLBot to EnhancedObs observation format.

This module converts RLBot's GameTickPacket to the observation format
expected by models trained with EnhancedObs (boost pads + game state).

Observation Format (89 dims):
- Ball: 9 dims (position, velocity, angular velocity)
- Self: 20 dims (position, velocity, forward, up, angular vel, boost, flags)
- Boost Pads: 34 dims (active status for all 34 pads)
- Game State: 6 dims (goal diff, time left, overtime, demo self, demo opp, padding)
- Opponent: 20 dims (same as self)
"""
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import math


class StateConverter:
    """
    Converts RLBot game state to EnhancedObs format.

    This produces 89-dimensional observations matching the training format:
    - Ball state (9 dims)
    - Self state (20 dims)
    - Boost pad states (34 dims)
    - Game state info (6 dims)
    - Opponent state (20 dims)
    """

    # Normalization constants
    POS_STD = 2300.0
    VEL_STD = 2300.0
    ANG_STD = math.pi

    # Boost pad locations (all 34 pads in standard order)
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
        """Initialize the state converter."""
        self.team = None
        self.previous_action = np.zeros(8, dtype=np.float32)
        print("StateConverter: Using EnhancedObs format (89 dims with boost pads)")

    def set_team(self, team: int):
        """Set the bot's team (0 = blue, 1 = orange)"""
        self.team = team

    def update_previous_action(self, action: np.ndarray):
        """Update the previous action cache"""
        self.previous_action = np.array(action, dtype=np.float32)

    def packet_to_observation(self, packet: GameTickPacket, index: int) -> np.ndarray:
        """
        Convert RLBot GameTickPacket to EnhancedObs format.

        Args:
            packet: RLBot GameTickPacket containing full game state
            index: Bot's player index

        Returns:
            np.ndarray: 89-dimensional observation array
        """
        if self.team is None:
            raise ValueError("Team must be set before converting observations")

        obs = []

        player = packet.game_cars[index]
        ball = packet.game_ball

        # Invert values if on orange team (RLGym convention)
        inv = -1 if self.team == 1 else 1

        # === BALL STATE (9 dims) ===
        obs.extend([
            ball.physics.location.x / self.POS_STD * inv,
            ball.physics.location.y / self.POS_STD * inv,
            ball.physics.location.z / self.POS_STD,
            ball.physics.velocity.x / self.VEL_STD * inv,
            ball.physics.velocity.y / self.VEL_STD * inv,
            ball.physics.velocity.z / self.VEL_STD,
            ball.physics.angular_velocity.x / self.ANG_STD,
            ball.physics.angular_velocity.y / self.ANG_STD,
            ball.physics.angular_velocity.z / self.ANG_STD,
        ])

        # === SELF STATE (20 dims) ===
        obs.extend(self._get_player_obs(player, inv))

        # === BOOST PADS (34 dims) ===
        # Get boost pad states from game info
        for i in range(34):
            if i < packet.num_boost:
                pad = packet.game_boosts[i]
                # 1.0 if active (can be picked up), 0.0 if inactive
                obs.append(1.0 if pad.is_active else 0.0)
            else:
                # Default to active if we don't have info
                obs.append(1.0)

        # === GAME STATE (6 dims) ===
        # Goal difference (from this player's perspective)
        game_info = packet.game_info
        blue_score = packet.teams[0].score if packet.num_teams > 0 else 0
        orange_score = packet.teams[1].score if packet.num_teams > 1 else 0

        if self.team == 0:  # Blue team
            goal_diff = blue_score - orange_score
        else:  # Orange team
            goal_diff = orange_score - blue_score
        obs.append(np.clip(goal_diff / 5.0, -1.0, 1.0))

        # Time remaining (normalized, 0-1)
        # RLBot provides seconds_elapsed and game_time_remaining
        if game_info.game_time_remaining > 0:
            # Assume 5-minute matches (300 seconds)
            time_left = np.clip(game_info.game_time_remaining / 300.0, 0.0, 1.0)
        else:
            time_left = 0.5  # Default if not available
        obs.append(time_left)

        # Is overtime
        is_overtime = 1.0 if game_info.is_overtime else 0.0
        obs.append(is_overtime)

        # Demo state for self
        demo_self = 1.0 if player.is_demolished else 0.0
        obs.append(demo_self)

        # Demo state for opponent (first opponent found)
        demo_opp = 0.0
        for i in range(packet.num_cars):
            if i != index:
                car = packet.game_cars[i]
                if car.team != self.team:
                    demo_opp = 1.0 if car.is_demolished else 0.0
                    break
        obs.append(demo_opp)

        # Padding (1 dim)
        obs.append(0.0)

        # === OPPONENT STATE (20 dims) ===
        opponent = None
        for i in range(packet.num_cars):
            if i != index:
                car = packet.game_cars[i]
                if car.team != self.team:
                    opponent = car
                    break

        if opponent is not None:
            obs.extend(self._get_player_obs(opponent, inv))
        else:
            # Pad with zeros if no opponent
            obs.extend([0.0] * 20)

        return np.array(obs, dtype=np.float32)

    def _get_player_obs(self, player, inv: int) -> list:
        """
        Get observation data for a player.

        Returns 20 dimensions:
        - Position (3)
        - Velocity (3)
        - Forward vector (3)
        - Up vector (3)
        - Angular velocity (3)
        - Boost (1)
        - Flags: on_ground, is_supersonic, has_jump, has_flip (4)
        """
        obs = []

        # Position
        obs.extend([
            player.physics.location.x / self.POS_STD * inv,
            player.physics.location.y / self.POS_STD * inv,
            player.physics.location.z / self.POS_STD,
        ])

        # Velocity
        obs.extend([
            player.physics.velocity.x / self.VEL_STD * inv,
            player.physics.velocity.y / self.VEL_STD * inv,
            player.physics.velocity.z / self.VEL_STD,
        ])

        # Forward vector
        forward = self._rotation_to_forward(player.physics.rotation)
        obs.extend([
            forward[0] * inv,
            forward[1] * inv,
            forward[2],
        ])

        # Up vector
        up = self._rotation_to_up(player.physics.rotation)
        obs.extend([
            up[0] * inv,
            up[1] * inv,
            up[2],
        ])

        # Angular velocity
        obs.extend([
            player.physics.angular_velocity.x / self.ANG_STD,
            player.physics.angular_velocity.y / self.ANG_STD,
            player.physics.angular_velocity.z / self.ANG_STD,
        ])

        # Boost (0-1)
        obs.append(player.boost / 100.0)

        # Flags
        obs.append(1.0 if player.has_wheel_contact else 0.0)  # on_ground
        obs.append(1.0 if player.is_super_sonic else 0.0)     # is_supersonic
        # has_jump: True if on ground or hasn't used jump yet
        has_jump = player.has_wheel_contact or not player.jumped
        obs.append(1.0 if has_jump else 0.0)
        # has_flip: True if hasn't double jumped and (on ground or has jumped)
        has_flip = not player.double_jumped and (player.has_wheel_contact or player.jumped)
        obs.append(1.0 if has_flip else 0.0)

        return obs

    @staticmethod
    def _rotation_to_forward(rotation) -> np.ndarray:
        """Convert RLBot rotation to forward vector"""
        pitch = rotation.pitch
        yaw = rotation.yaw

        forward = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch)
        ])
        return forward

    @staticmethod
    def _rotation_to_up(rotation) -> np.ndarray:
        """Convert RLBot rotation to up vector"""
        pitch = rotation.pitch
        yaw = rotation.yaw
        roll = rotation.roll

        up = np.array([
            -math.sin(pitch) * math.cos(yaw) * math.cos(roll) - math.sin(yaw) * math.sin(roll),
            -math.sin(pitch) * math.sin(yaw) * math.cos(roll) + math.cos(yaw) * math.sin(roll),
            math.cos(pitch) * math.cos(roll)
        ])
        return up


def model_action_to_controller(action: np.ndarray) -> SimpleControllerState:
    """
    Convert model's 8D continuous action to RLBot controller state.

    Action format: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    All values are continuous in range [-1, 1]

    Args:
        action: 8D numpy array from model

    Returns:
        SimpleControllerState for RLBot
    """
    controller = SimpleControllerState()

    # Continuous controls
    controller.throttle = float(np.clip(action[0], -1, 1))
    controller.steer = float(np.clip(action[1], -1, 1))
    controller.pitch = float(np.clip(action[2], -1, 1))
    controller.yaw = float(np.clip(action[3], -1, 1))
    controller.roll = float(np.clip(action[4], -1, 1))

    # Binary controls (threshold at 0)
    controller.jump = bool(action[5] > 0)
    controller.boost = bool(action[6] > 0)
    controller.handbrake = bool(action[7] > 0)

    return controller
