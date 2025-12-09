"""
State converter for RLBot to RLGym-Sim observation format.

This module uses rlgym-compat for accurate conversion between RLBot's
GameTickPacket and the observation format expected by models trained with RLGym-Sim.

IMPORTANT: The observation format MUST match exactly what was used during training.
Using rlgym-compat ensures this compatibility.
"""
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
import math

# Try to import rlgym-compat for proper observation conversion
try:
    from rlgym_compat import GameState
    from rlgym_sim.utils.obs_builders import DefaultObs
    RLGYM_COMPAT_AVAILABLE = True
except ImportError:
    RLGYM_COMPAT_AVAILABLE = False
    print("WARNING: rlgym-compat not available. Using fallback manual conversion.")
    print("For best results, install rlgym-compat: pip install rlgym-compat")


class StateConverter:
    """
    Converts RLBot game state to RLGym-Sim DefaultObs format.

    This class provides two modes of operation:
    1. If rlgym-compat is installed: Uses the official conversion for exact compatibility
    2. Fallback mode: Manual conversion (may have slight differences)

    The DefaultObs format from RLGym-Sim includes:
    - Ball data (position, velocity, angular velocity)
    - Player car data (position, velocity, rotation vectors, angular velocity, boost, flags)
    - Teammate data (if any)
    - Opponent data (if any)

    Note: RLGym's DefaultObs may also include:
    - Previous action (depending on version)
    - Boost pad states (depending on version)

    Always verify your observation dimensions match between training and gameplay!
    """

    # Normalization constants (RLGym-Sim standard values)
    POS_STD = 2300
    ANG_STD = math.pi
    VEL_STD = 2300

    def __init__(self, use_compat=True, legacy_mode=False):
        """
        Initialize the state converter.

        Args:
            use_compat: Whether to use rlgym-compat if available (recommended for new training)
            legacy_mode: If True, use the original observation format for compatibility
                        with models trained before the fix. This uses the old player flags:
                        (has_wheel_contact, is_super_sonic, jumped, double_jumped)
                        instead of DefaultObs format (on_ground, has_flip, is_demoed).
                        Set this to True if testing with a model trained with the old code.
        """
        self.team = None
        self.legacy_mode = legacy_mode

        # In legacy mode, don't use rlgym-compat even if available
        if legacy_mode:
            self.use_compat = False
            print("StateConverter: LEGACY MODE - using original observation format")
            print("  (for compatibility with models trained before the fix)")
        else:
            self.use_compat = use_compat and RLGYM_COMPAT_AVAILABLE

        # Initialize rlgym-compat components if available and not in legacy mode
        if self.use_compat:
            self.game_state = GameState()
            self.obs_builder = DefaultObs()
            print("StateConverter: Using rlgym-compat for observation conversion")
        else:
            self.game_state = None
            self.obs_builder = None
            if not legacy_mode:
                print("StateConverter: Using manual observation conversion")

        # Cache for previous action (some DefaultObs versions include this)
        self.previous_action = np.zeros(8, dtype=np.float32)

    def set_team(self, team: int):
        """Set the bot's team (0 = blue, 1 = orange)"""
        self.team = team

    def update_previous_action(self, action: np.ndarray):
        """Update the previous action cache (for obs builders that use it)"""
        self.previous_action = np.array(action, dtype=np.float32)

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

        if self.use_compat:
            return self._convert_with_compat(packet, index)
        else:
            return self._convert_manual(packet, index)

    def _convert_with_compat(self, packet: GameTickPacket, index: int) -> np.ndarray:
        """
        Convert using rlgym-compat (most accurate method).

        This uses the official RLGym compatibility layer to ensure
        observations match exactly what was used during training.
        """
        # Decode the RLBot packet into RLGym GameState format
        self.game_state.decode(packet)

        # Get the player object
        player = self.game_state.players[index]

        # Build observation using the same DefaultObs used during training
        obs = self.obs_builder.build_obs(
            player=player,
            state=self.game_state,
            previous_action=self.previous_action
        )

        return np.array(obs, dtype=np.float32).flatten()

    def _convert_manual(self, packet: GameTickPacket, index: int) -> np.ndarray:
        """
        Manual conversion fallback (may have slight differences from RLGym).

        WARNING: This method attempts to replicate RLGym's DefaultObs format
        manually. Small differences in normalization, ordering, or rotation
        calculations could cause the model to underperform.

        Consider installing rlgym-compat for exact compatibility.
        """
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

        # Player rotation (as forward and up vectors)
        forward = self._rotation_to_forward(player.physics.rotation)
        up = self._rotation_to_up(player.physics.rotation)

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

        # Player flags - format depends on legacy_mode
        if self.legacy_mode:
            # LEGACY FORMAT: Original observation format (for models trained before fix)
            # Uses: has_wheel_contact, is_super_sonic, jumped, double_jumped
            obs.append(1.0 if player.has_wheel_contact else 0.0)
            obs.append(1.0 if player.is_super_sonic else 0.0)
            obs.append(1.0 if player.jumped else 0.0)
            obs.append(1.0 if player.double_jumped else 0.0)
        else:
            # NEW FORMAT: Matches RLGym DefaultObs
            # Uses: on_ground, has_flip, is_demoed
            on_ground = 1.0 if player.has_wheel_contact else 0.0
            # has_flip: True if player hasn't used their flip yet after jumping
            has_flip = 1.0 if (not player.double_jumped and (player.has_wheel_contact or player.jumped)) else 0.0
            is_demoed = 1.0 if player.is_demolished else 0.0
            obs.append(on_ground)
            obs.append(has_flip)
            obs.append(is_demoed)

        # === TEAMMATE AND OPPONENT DATA ===
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
        """Get observation data for another car (teammate or opponent)"""
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

        # Flags - format depends on legacy_mode
        if self.legacy_mode:
            # LEGACY FORMAT
            obs.append(1.0 if car.has_wheel_contact else 0.0)
            obs.append(1.0 if car.is_super_sonic else 0.0)
            obs.append(1.0 if car.jumped else 0.0)
            obs.append(1.0 if car.double_jumped else 0.0)
        else:
            # NEW FORMAT (DefaultObs)
            on_ground = 1.0 if car.has_wheel_contact else 0.0
            has_flip = 1.0 if (not car.double_jumped and (car.has_wheel_contact or car.jumped)) else 0.0
            is_demoed = 1.0 if car.is_demolished else 0.0
            obs.append(on_ground)
            obs.append(has_flip)
            obs.append(is_demoed)

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
