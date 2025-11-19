# Vexril-RocketLeague

A Reinforcement Learning bot for Rocket League built with RLGym and Stable-Baselines3.

## Overview

This project implements a baseline RL agent that learns to play Rocket League using the Proximal Policy Optimization (PPO) algorithm. The bot learns from scratch in a simulated environment provided by RLGym.

## Features

- **PPO Training**: Uses the PPO algorithm from Stable-Baselines3
- **Custom Reward Function**: Rewards the bot for:
  - Moving towards the ball
  - Touching the ball
  - Scoring goals
  - Moving the ball towards opponent's goal
- **Varied Training Scenarios**: Custom state setter creates diverse initial conditions
- **Configurable Training**: YAML-based configuration for easy hyperparameter tuning
- **Evaluation Script**: Test and benchmark trained models

## Project Structure

```
Vexril-RocketLeague/
├── configs/
│   └── training_config.yaml    # Training configuration
├── src/
│   ├── agents/
│   │   └── rl_bot.py          # Bot agent classes
│   ├── rewards/
│   │   └── custom_reward.py   # Custom reward function
│   ├── state_setters/
│   │   └── custom_state_setter.py  # Custom state initialization
│   └── utils/                  # Utility functions
├── models/                     # Saved models (created during training)
├── logs/                       # TensorBoard logs (created during training)
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
└── requirements.txt            # Python dependencies
```

## Prerequisites

- Python 3.8 or higher
- Rocket League (installed via Epic Games or Steam)
- Windows OS (RLGym requires Windows for Rocket League integration)

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Vexril-RocketLeague
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install RLGym**
RLGym requires additional setup to connect to Rocket League. Follow the official [RLGym installation guide](https://rlgym.github.io/).

## Usage

### Training

To start training the bot:

```bash
python train.py
```

The training script will:
- Create the RLGym environment
- Initialize or load a PPO model
- Train for the specified number of timesteps
- Save checkpoints periodically
- Log training metrics to TensorBoard

**Monitor training progress:**
```bash
tensorboard --logdir=./logs
```
Then open http://localhost:6006 in your browser.

### Configuration

Edit `configs/training_config.yaml` to adjust:
- Environment settings (game speed, team size, etc.)
- PPO hyperparameters (learning rate, batch size, etc.)
- Training duration and checkpoint frequency

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model models/rl_bot_final.zip --episodes 10
```

Options:
- `--model`: Path to the trained model (default: `models/rl_bot_final.zip`)
- `--episodes`: Number of episodes to evaluate (default: 10)
- `--render`: Render the game during evaluation (requires proper setup)

### Resume Training

To resume training from a checkpoint:

1. Set `resume_training: true` in `configs/training_config.yaml`
2. Ensure the model exists at `models/rl_bot_latest.zip`
3. Run `python train.py`

## Customization

### Custom Reward Function

Modify `src/rewards/custom_reward.py` to change how the bot is rewarded. The current implementation rewards:
- Ball touches
- Proximity to ball
- Velocity towards ball
- Ball velocity towards opponent's goal
- Goals scored

### Custom State Setter

Modify `src/state_setters/custom_state_setter.py` to change initial game states. This affects the diversity of training scenarios.

### Hyperparameter Tuning

Common hyperparameters to tune in `configs/training_config.yaml`:
- `learning_rate`: Higher = faster learning but less stable
- `n_steps`: More steps = more diverse experience per update
- `batch_size`: Larger = more stable updates but slower
- `ent_coef`: Higher = more exploration

## Training Tips

1. **Start with default settings**: The baseline configuration should work well for initial training
2. **Monitor TensorBoard**: Watch for reward trends and policy loss
3. **Be patient**: Training can take several hours to days depending on your hardware
4. **Curriculum learning**: Start with simpler scenarios and gradually increase complexity
5. **Save checkpoints**: Regular checkpoints allow you to recover from training issues

## Common Issues

### RLGym Connection Issues
- Ensure Rocket League is closed before starting training
- RLGym will launch Rocket League automatically
- Check firewall settings if connection fails

### Out of Memory
- Reduce `n_steps` or `batch_size` in the config
- Close other applications to free up RAM

### Slow Training
- Increase `game_speed` in the config (higher = faster simulation)
- Reduce `tick_skip` (fewer physics steps per action)
- Use a GPU for faster neural network training

## Performance Expectations

With default settings:
- **Training time**: 4-12 hours for basic ball-chasing behavior
- **Hardware**: GPU recommended for faster training
- **Timesteps**: ~1-5 million for basic skills, 10-50 million for advanced play

## Next Steps

After establishing a baseline:
1. Implement more sophisticated reward functions
2. Add aerial training scenarios
3. Implement self-play for competitive training
4. Add demonstrations/imitation learning
5. Experiment with different algorithms (SAC, TD3, etc.)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Resources

- [RLGym Documentation](https://rlgym.github.io/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

## License

This project is open source and available under the MIT License.

## Acknowledgments

- RLGym team for the excellent Rocket League environment
- Stable-Baselines3 for the RL implementations
- Rocket League community for inspiration
