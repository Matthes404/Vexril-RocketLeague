# Setting Up Collision Meshes for RLGym-Sim

RLGym-Sim requires collision mesh files from Rocket League to accurately simulate the arena physics. These files cannot be redistributed, so you must extract them from your own copy of Rocket League.

## Prerequisites

- **Rocket League installed** (via Epic Games or Steam)
- **Windows OS** (the dumper tool is Windows-only)

## Step-by-Step Instructions

### Option 1: Using RLArenaCollisionDumper (Recommended)

1. **Download the Asset Dumper**
   - Go to: https://github.com/ZealanL/RLArenaCollisionDumper/releases/tag/v1.0.0
   - Download `RLArenaCollisionDumper.exe`

2. **Run the Dumper**
   - Launch Rocket League
   - Go to Free Play mode
   - Run `RLArenaCollisionDumper.exe`
   - The tool will create a `collision_meshes` folder in the same directory

3. **Move the Files**
   - Copy the entire `collision_meshes` folder to your project directory:
     ```
     Vexril-RocketLeague/
     ├── collision_meshes/    <-- Place it here
     ├── configs/
     ├── src/
     ├── train.py
     └── ...
     ```

4. **Verify Setup**
   - The folder structure should look like:
     ```
     collision_meshes/
     ├── soccar.cmf
     └── ... (other mesh files)
     ```

### Option 2: If You Don't Have Rocket League

If you don't own Rocket League, you'll need to either:
- Purchase the game to extract the collision meshes
- Ask someone who owns the game to extract and share them with you (for personal use only)

**Note**: We cannot provide pre-extracted collision meshes due to copyright restrictions.

## Troubleshooting

### "No arena meshes found" Error
- Make sure the `collision_meshes` folder is in the **same directory** as `train.py`
- Verify that the folder contains `.cmf` files (collision mesh format)
- Check that the folder is named exactly `collision_meshes` (lowercase, underscore)

### RLArenaCollisionDumper Won't Run
- Ensure Rocket League is running and in Free Play mode
- Try running the dumper as Administrator
- Check that your antivirus isn't blocking it

### Linux/macOS Users
- The collision dumper only works on Windows
- You can:
  1. Use a Windows machine or VM to dump the meshes
  2. Use Wine to run the Windows executable (may not work)
  3. Ask a friend with Windows to dump the meshes for you

## After Setup

Once the collision meshes are in place, you can run:

```bash
python train.py
```

The error about missing collision meshes should be resolved!

## Important Notes

- The collision meshes are copyrighted by Psyonix/Epic Games
- Only extract them if you own a legitimate copy of Rocket League
- Do not redistribute the extracted files publicly
- The `collision_meshes` folder is ignored by git (see `.gitignore`)

## Need Help?

If you continue to have issues:
1. Check the [RLArenaCollisionDumper README](https://github.com/ZealanL/RLArenaCollisionDumper/blob/main/README.md)
2. Visit the [RLGym Discord](https://discord.gg/rlgym) for community support
3. Open an issue on this repository
