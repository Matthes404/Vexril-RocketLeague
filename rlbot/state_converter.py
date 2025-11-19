"""
State converter for RLBot to RLGym-Sim observation format.
Converts RLBot's GameTickPacket to the observation format expected by models trained with RLGym-Sim.
"""
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import math


class StateConverter:
    """
    Converts RLBot game state to RLGym-Sim DefaultObs format.

    The DefaultObs format from RLGym-Sim includes:
    - Player car data (position, velocity, rotation, angular velocity, boost, etc.)
    - Ball data (position, velocity, angular velocity)
    - Teammate data (if any)
    - Opponent data (if any)
    """

    # Normalization constants (RLGym-Sim standard values)
    POS_STD = 2300
    ANG_STD = math.pi
    VEL_STD = 2300

    def __init__(self):
        self.team = None  # Will be set when bot initializes

    def set_team(self, team: int):
        """Set the bot's team (0 = blue, 1 = orange)"""
        self.team = team

    def packet_to_observation(self, packet: GameTickPacket, index: int) -> np.ndarray:
        """
        Convert RLBot GameTickPacket to RLGym observation format.

        Args:
            packet: RLBot GameTickPacket containing full game state
            index: Bot's player index

        Returns:
            np.ndarray: Flattened observation matching RLGym-Sim DefaultObs format
        """
        if self.team is None:
            raise ValueError("Team must be set before converting observations")

        obs = []

        player = packet.game_cars[index]
        ball = packet.game_ball

        # Invert values if on orange team (RLGym convention)
        inv = -1 if self.team == 1 else 1

        # === BALL STATE ===
        # Ball position (normalized)
        obs.extend([
            ball.physics.location.x / self.POS_STD * inv,
            ball.physics.location.y / self.POS_STD * inv,
            ball.physics.location.z / self.POS_STD
        ])

        # Ball velocity (normalized)
        obs.extend([
            ball.physics.velocity.x / self.VEL_STD * inv,
            ball.physics.velocity.y / self.VEL_STD * inv,
            ball.physics.velocity.z / self.VEL_STD
        ])

        # Ball angular velocity (normalized)
        obs.extend([
            ball.physics.angular_velocity.x / self.ANG_STD,
            ball.physics.angular_velocity.y / self.ANG_STD,
            ball.physics.angular_velocity.z / self.ANG_STD
        ])

        # === PLAYER STATE ===
        # Player position (normalized)
        obs.extend([
            player.physics.location.x / self.POS_STD * inv,
            player.physics.location.y / self.POS_STD * inv,
            player.physics.location.z / self.POS_STD
        ])

        # Player velocity (normalized)
        obs.extend([
            player.physics.velocity.x / self.VEL_STD * inv,
            player.physics.velocity.y / self.VEL_STD * inv,
            player.physics.velocity.z / self.VEL_STD
        ])

        # Player rotation (as forward, up, and right vectors)
        forward = self._rotation_to_forward(player.physics.rotation)
        up = self._rotation_to_up(player.physics.rotation)
        right = self._rotation_to_right(player.physics.rotation)

        obs.extend([
            forward[0] * inv,
            forward[1] * inv,
            forward[2]
        ])

        obs.extend([
            up[0] * inv,
            up[1] * inv,
            up[2]
        ])

        # Player angular velocity (normalized)
        obs.extend([
            player.physics.angular_velocity.x / self.ANG_STD,
            player.physics.angular_velocity.y / self.ANG_STD,
            player.physics.angular_velocity.z / self.ANG_STD
        ])

        # Player boost amount (0-1)
        obs.append(player.boost / 100.0)

        # Player flags
        obs.append(1.0 if player.has_wheel_contact else 0.0)
        obs.append(1.0 if player.is_super_sonic else 0.0)
        obs.append(1.0 if player.jumped else 0.0)
        obs.append(1.0 if player.double_jumped else 0.0)

        # === TEAMMATE AND OPPONENT DATA ===
        # Get teammates and opponents
        teammates = []
        opponents = []

        for i in range(packet.num_cars):
            if i == index:
                continue
            car = packet.game_cars[i]
            if car.team == self.team:
                teammates.append((i, car))
            else:
                opponents.append((i, car))

        # Add teammate data (if any)
        for _, car in teammates:
            obs.extend(self._get_car_obs(car, inv))

        # Add opponent data (if any)
        for _, car in opponents:
            obs.extend(self._get_car_obs(car, inv))

        return np.array(obs, dtype=np.float32)

    def _get_car_obs(self, car, inv: int) -> list:
        """Get observation data for another car"""
        obs = []

        # Position
        obs.extend([
            car.physics.location.x / self.POS_STD * inv,
            car.physics.location.y / self.POS_STD * inv,
            car.physics.location.z / self.POS_STD
        ])

        # Velocity
        obs.extend([
            car.physics.velocity.x / self.VEL_STD * inv,
            car.physics.velocity.y / self.VEL_STD * inv,
            car.physics.velocity.z / self.VEL_STD
        ])

        # Rotation vectors
        forward = self._rotation_to_forward(car.physics.rotation)
        up = self._rotation_to_up(car.physics.rotation)

        obs.extend([
            forward[0] * inv,
            forward[1] * inv,
            forward[2]
        ])

        obs.extend([
            up[0] * inv,
            up[1] * inv,
            up[2]
        ])

        # Angular velocity
        obs.extend([
            car.physics.angular_velocity.x / self.ANG_STD,
            car.physics.angular_velocity.y / self.ANG_STD,
            car.physics.angular_velocity.z / self.ANG_STD
        ])

        # Boost
        obs.append(car.boost / 100.0)

        # Flags
        obs.append(1.0 if car.has_wheel_contact else 0.0)
        obs.append(1.0 if car.is_super_sonic else 0.0)
        obs.append(1.0 if car.jumped else 0.0)
        obs.append(1.0 if car.double_jumped else 0.0)

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

    @staticmethod
    def _rotation_to_right(rotation) -> np.ndarray:
        """Convert RLBot rotation to right vector"""
        pitch = rotation.pitch
        yaw = rotation.yaw
        roll = rotation.roll

        right = np.array([
            math.sin(pitch) * math.cos(yaw) * math.sin(roll) - math.sin(yaw) * math.cos(roll),
            math.sin(pitch) * math.sin(yaw) * math.sin(roll) + math.cos(yaw) * math.cos(roll),
            -math.cos(pitch) * math.sin(roll)
        ])
        return right


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

    # Continuous controls (already in correct range)
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
