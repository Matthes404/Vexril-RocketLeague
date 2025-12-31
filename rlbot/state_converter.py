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

    # Supersonic threshold (same as EnhancedObs)
    SUPERSONIC_THRESHOLD = 2200.0

    # Boost pad locations in RLGym-Sim order (all 34 pads)
    # This is the order RLGym-Sim uses internally for state.boost_pads
    # Format: (x, y, z, is_big)
    BOOST_PADS_RLGYM_ORDER = [
        # These are ordered by RLGym-Sim's internal indexing
        # Small pads first, then big pads, roughly sorted by position
        (0.0, -4240.0, 70.0, False),      # 0
        (-1792.0, -4184.0, 70.0, False),  # 1
        (1792.0, -4184.0, 70.0, False),   # 2
        (-940.0, -3308.0, 70.0, False),   # 3
        (940.0, -3308.0, 70.0, False),    # 4
        (0.0, -2816.0, 70.0, False),      # 5
        (-3584.0, -2484.0, 70.0, False),  # 6
        (3584.0, -2484.0, 70.0, False),   # 7
        (-1788.0, -2300.0, 70.0, False),  # 8
        (1788.0, -2300.0, 70.0, False),   # 9
        (-2048.0, -1036.0, 70.0, False),  # 10
        (0.0, -1024.0, 70.0, False),      # 11
        (2048.0, -1036.0, 70.0, False),   # 12
        (-1024.0, 0.0, 70.0, False),      # 13
        (1024.0, 0.0, 70.0, False),       # 14
        (-2048.0, 1036.0, 70.0, False),   # 15
        (0.0, 1024.0, 70.0, False),       # 16
        (2048.0, 1036.0, 70.0, False),    # 17
        (-1788.0, 2300.0, 70.0, False),   # 18
        (1788.0, 2300.0, 70.0, False),    # 19
        (-3584.0, 2484.0, 70.0, False),   # 20
        (3584.0, 2484.0, 70.0, False),    # 21
        (0.0, 2816.0, 70.0, False),       # 22
        (-940.0, 3308.0, 70.0, False),    # 23
        (940.0, 3308.0, 70.0, False),     # 24
        (-1792.0, 4184.0, 70.0, False),   # 25
        (1792.0, 4184.0, 70.0, False),    # 26
        (0.0, 4240.0, 70.0, False),       # 27
        # Big boost pads (6 total)
        (-3072.0, -4096.0, 73.0, True),   # 28
        (3072.0, -4096.0, 73.0, True),    # 29
        (-3584.0, 0.0, 73.0, True),       # 30
        (3584.0, 0.0, 73.0, True),        # 31
        (-3072.0, 4096.0, 73.0, True),    # 32
        (3072.0, 4096.0, 73.0, True),     # 33
    ]

    def __init__(self):
        """Initialize the state converter."""
        self.team = None
        self.previous_action = np.zeros(8, dtype=np.float32)
        # Mapping from RLBot boost index to RLGym-Sim boost index
        # Built on first packet when we have boost pad positions
        self.boost_pad_mapping = None
        self.boost_mapping_built = False
        print("StateConverter: Using EnhancedObs format (89 dims with boost pads)")

    def set_team(self, team: int):
        """Set the bot's team (0 = blue, 1 = orange)"""
        self.team = team

    def update_previous_action(self, action: np.ndarray):
        """Update the previous action cache"""
        self.previous_action = np.array(action, dtype=np.float32)

    def _build_boost_pad_mapping(self, packet: GameTickPacket):
        """
        Build a mapping from RLBot boost pad indices to RLGym-Sim indices.

        RLBot and RLGym-Sim may order boost pads differently, so we need to
        match them by position to ensure consistency.
        """
        if self.boost_mapping_built:
            return

        self.boost_pad_mapping = [None] * 34

        # For each RLGym-Sim boost pad position, find the closest RLBot boost pad
        for rlgym_idx, (rx, ry, rz, _) in enumerate(self.BOOST_PADS_RLGYM_ORDER):
            best_rlbot_idx = None
            best_dist = float('inf')

            for rlbot_idx in range(min(packet.num_boost, 34)):
                pad = packet.game_boosts[rlbot_idx]
                bx = pad.location.x
                by = pad.location.y
                bz = pad.location.z

                dist = math.sqrt((rx - bx)**2 + (ry - by)**2 + (rz - bz)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_rlbot_idx = rlbot_idx

            # Should be very close (within a few units)
            if best_dist < 50.0:
                self.boost_pad_mapping[rlgym_idx] = best_rlbot_idx
            else:
                # No match found, default to same index
                self.boost_pad_mapping[rlgym_idx] = rlgym_idx

        self.boost_mapping_built = True
        print(f"StateConverter: Built boost pad mapping (34 pads)")

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

        # Build boost pad mapping on first call
        if not self.boost_mapping_built:
            self._build_boost_pad_mapping(packet)

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
        # Get boost pad states using the RLGym-Sim ordering
        for rlgym_idx in range(34):
            rlbot_idx = self.boost_pad_mapping[rlgym_idx] if self.boost_pad_mapping else rlgym_idx
            if rlbot_idx is not None and rlbot_idx < packet.num_boost:
                pad = packet.game_boosts[rlbot_idx]
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

        Returns 20 dimensions (matching EnhancedObs exactly):
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
        vel_x = player.physics.velocity.x
        vel_y = player.physics.velocity.y
        vel_z = player.physics.velocity.z
        obs.extend([
            vel_x / self.VEL_STD * inv,
            vel_y / self.VEL_STD * inv,
            vel_z / self.VEL_STD,
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

        # === FLAGS (must match EnhancedObs exactly!) ===

        # on_ground: matches player.on_ground in RLGym-Sim
        obs.append(1.0 if player.has_wheel_contact else 0.0)

        # is_supersonic: Calculate from velocity (same as EnhancedObs)
        # EnhancedObs uses: speed >= 2200.0
        speed = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        obs.append(1.0 if speed >= self.SUPERSONIC_THRESHOLD else 0.0)

        # has_jump: In RLGym-Sim, this is True if player can still first-jump
        # - On ground: always True
        # - In air without having jumped: True (briefly after leaving ground)
        # - After jumping: False
        # RLBot approximation: on ground OR hasn't pressed jump yet
        has_jump = player.has_wheel_contact or not player.jumped
        obs.append(1.0 if has_jump else 0.0)

        # has_flip: In RLGym-Sim, this is True if player can still flip/dodge
        # - On ground: False (must jump first to flip)
        # - In air after jumping, before flip expires: True
        # - After double jumping/flipping: False
        # RLBot approximation: in air (or jumped) AND hasn't double jumped
        # Note: We can't track the 1.5s flip timer, so this is approximate
        has_flip = (not player.has_wheel_contact or player.jumped) and not player.double_jumped
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
