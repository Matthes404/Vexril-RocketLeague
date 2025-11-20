# Vexril Bot - RLBot Integration

This folder contains everything needed to run your trained RLGym-Sim bot in **actual Rocket League** using RLBotGUI.

## Overview

The bot trained with RLGym-Sim (in simulation) can be deployed to play in real Rocket League on Windows. This integration bridges the gap between the training environment and the actual game.

## Prerequisites

### Windows Machine Required
- **Rocket League** (Epic Games or Steam version)
- **Python 3.7-3.10** (RLBot currently supports up to Python 3.10)
- **RLBotGUI** installed

### Trained Model Required
You need a trained model file (`.zip`) in the `models/` directory. The bot will automatically look for:
1. `models/rl_bot_final.zip` (final trained model)
2. `models/rl_bot_latest.zip` (latest checkpoint)
3. Any checkpoint file matching `models/rl_bot_*.zip`

**IMPORTANT:** You also need the `models/rl_bot_vecnormalize.pkl` file! This file contains observation normalization statistics that are critical for the bot to work correctly. Without it, the bot will receive observations in the wrong scale and perform very poorly. This file is automatically created during training.

## Installation on Windows

### 1. Install RLBotGUI

Download and install RLBotGUI from [rlbot.org](https://rlbot.org/):
- Visit https://rlbot.org/
- Download the installer for Windows
- Run the installer and follow the setup instructions
- This will install RLBot and set up your Rocket League integration

### 2. Install Python Dependencies

Open a terminal/command prompt in the `rlbot/` directory and run:

```bash
pip install -r requirements.txt
```

This installs:
- `rlbot` - The RLBot framework
- `stable-baselines3` - For loading your trained PPO model
- `torch` - PyTorch (required by Stable-Baselines3)
- `numpy` - Numerical operations

### 3. Copy Your Trained Model

Make sure your trained model is in the `models/` directory (one level up from this folder):

```
Vexril-RocketLeague/
├── models/
│   ├── rl_bot_final.zip           ← Your trained model
│   ├── rl_bot_latest.zip          ← Or latest checkpoint
│   └── rl_bot_vecnormalize.pkl    ← CRITICAL: Observation normalization stats
└── rlbot/
    ├── vexril_bot.py
    ├── bot.cfg
    ├── state_converter.py
    └── ...
```

## Usage

### Method 1: Using RLBotGUI (Recommended)

1. **Open RLBotGUI**
2. **Add the bot:**
   - Click "Add" → "Add bot"
   - Navigate to this `rlbot/` folder
   - Select `bot.cfg`
3. **Start a match:**
   - Add your bot to a team
   - Add opponents (other bots or leave slots for human players)
   - Click "Start Match"
4. **Play!**
   - Rocket League will launch automatically
   - Your bot will join the match
   - You can play alongside or against your bot

### Method 2: Command Line

You can also run matches via command line using RLBot's Python API:

```python
from rlbot.setup_manager import SetupManager

setup_manager = SetupManager()
setup_manager.load_config(config_location='./bot.cfg')
setup_manager.launch_ball_prediction()
setup_manager.launch_quick_chat_manager()
setup_manager.launch_bot_processes()
setup_manager.start_match()
```

## How It Works

### Architecture

```
Rocket League Game
        ↓ (game state via RLBot API)
   GameTickPacket
        ↓
   StateConverter.packet_to_observation()
        ↓ (converts to RLGym-Sim DefaultObs format: 49 dims for 1v1)
   Raw Observation [49 values]
        ↓
   VecNormalize.normalize_obs()
        ↓ (normalize using training stats - CRITICAL!)
   Normalized Observation
        ↓
   PPO Model (trained in simulation)
        ↓ (8D continuous action)
   model_action_to_controller()
        ↓ (converts to RLBot controller)
   SimpleControllerState
        ↓ (sent back to game)
Rocket League Game
```

### Key Technical Details

**Observation Format (DefaultObs):**
- For 1v1: 49-dimensional vector
  - Ball: 9 values (position, velocity, angular velocity)
  - Player: 20 values (position, velocity, rotation vectors, angular velocity, boost, flags)
  - Opponent: 20 values (same as player)
- For 2v2: 89 dimensions (1 teammate + 2 opponents)
- For 3v3: 129 dimensions (2 teammates + 3 opponents)

**Observation Normalization:**
- Position values normalized by 2300 (field size)
- Velocity values normalized by 2300 (max speed)
- Angular velocity normalized by π (rotation rate)
- Team inversion: Orange team observations are inverted for symmetry

**Action Space:**
- 8-dimensional continuous: `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`
- Continuous values in [-1, 1] range
- Binary controls (jump, boost, handbrake) thresholded at 0

### Key Components

1. **`vexril_bot.py`** - Main bot class
   - Inherits from RLBot's `BaseAgent`
   - Loads your trained PPO model
   - Gets game state every tick and returns actions

2. **`state_converter.py`** - State conversion
   - `StateConverter`: Converts RLBot's `GameTickPacket` to RLGym-Sim observation format
   - `model_action_to_controller()`: Converts model's 8D action to RLBot controller

3. **`bot.cfg`** - Bot configuration
   - Tells RLBot where to find your bot
   - Sets bot name and metadata

4. **`appearance.cfg`** - Visual customization
   - Car body, colors, wheels, boost, etc.
   - Separate configurations for blue and orange teams

## Customization

### Changing Bot Appearance

Edit `appearance.cfg` to customize how your bot looks in-game:
- Car body (`car_id`)
- Colors (`team_color_id`, `custom_color_id`)
- Wheels (`wheels_id`)
- Boost trail (`boost_id`)
- Goal explosion (`goal_explosion_id`)

Refer to [RLBot customization docs](https://github.com/RLBot/RLBot/wiki/Bot-Customization) for available IDs.

### Using a Different Model

To use a specific model checkpoint, modify `vexril_bot.py` line 48-63 to point to your desired model file:

```python
# Example: Force loading a specific checkpoint
self.model_path = models_dir / 'rl_bot_400000_steps.zip'
```

### Adjusting Bot Behavior

The bot uses **deterministic** action selection by default (line 93):
```python
action, _ = self.model.predict(observation, deterministic=True)
```

For more exploratory/random behavior, change to:
```python
action, _ = self.model.predict(observation, deterministic=False)
```

## Troubleshooting

### "No trained model found"
- Make sure you've trained a model using `train.py` first
- Check that the `.zip` file exists in `models/` directory
- Verify the file path in the console output

### "VecNormalize file not found" warning
- This is a CRITICAL warning - the bot will perform very poorly without normalization
- Make sure `models/rl_bot_vecnormalize.pkl` exists
- This file is created automatically during training
- If missing, you need to retrain or copy it from your training machine

### "Failed to load model"
- Ensure compatible versions of `stable-baselines3` and `torch`
- Check that the model file isn't corrupted
- Try loading the model manually in Python to see detailed error

### Bot doesn't move
- Check RLBotGUI console for error messages
- Verify that Rocket League is running in windowed or borderless mode
- Make sure RLBot framework is properly installed

### "Module not found" errors
- Install all dependencies: `pip install -r requirements.txt`
- Install main project requirements: `pip install -r ../requirements.txt`
- Ensure Python version is between 3.7 and 3.10

### Observation shape mismatch
- The bot was trained with a specific observation format (DefaultObs)
- If you modified `obs_builder` in training, update `state_converter.py` accordingly
- Check that the number of players in the match matches training conditions:
  - 1v1: 49 dimensions
  - 2v2: 89 dimensions
  - 3v3: 129 dimensions
- The bot currently expects observations for 1v1 (or will adapt to the number of cars detected)

## Performance Notes

- The bot runs inference on CPU by default (fast enough for real-time play)
- Each decision takes ~1-10ms depending on your hardware
- Rocket League runs at 120 ticks/second, the bot updates at the same rate

## Development Tips

### Testing State Conversion

You can test the state converter without launching a full match:

```python
from state_converter import StateConverter
converter = StateConverter()
converter.set_team(0)  # Blue team

# In a match, check observation dimensions
obs = converter.packet_to_observation(packet, player_index)
print(f"Observation shape: {obs.shape}")
```

### Debugging

Enable debug logging in `vexril_bot.py`:

```python
def get_output(self, packet):
    observation = self.state_converter.packet_to_observation(packet, self.index)
    self.logger.info(f"Obs: {observation[:10]}...")  # Log first 10 values
    action, _ = self.model.predict(observation, deterministic=True)
    self.logger.info(f"Action: {action}")
    # ...
```

### Monitoring Performance

RLBotGUI shows:
- Bot FPS (should be 120)
- Tick processing time
- Console output and errors

## Next Steps

1. **Train a better model** - More training steps = better performance
2. **Tune rewards** - Adjust `src/rewards/custom_reward.py` for desired behaviors
3. **Test in different scenarios** - 1v1, 2v2, 3v3, custom maps
4. **Compete** - Join RLBot tournaments and competitions!

## Resources

- [RLBot Discord](https://discord.gg/rlbot) - Get help from the community
- [RLBot Wiki](https://github.com/RLBot/RLBot/wiki) - Official documentation
- [RLGym-Sim Docs](https://rlgym-sim.readthedocs.io/) - Training framework docs
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/) - PPO algorithm docs

## License

Same as main project (see root LICENSE file).

---

**Good luck and have fun watching your bot play in real Rocket League! 🚗⚽**
