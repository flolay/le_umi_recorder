# UMI Recorder

Record robot teleoperation demonstrations using Meta Quest 3 controllers and webcams in LeRobot format.

## Architecture

```
Quest 3 Browser ──WebSocket──> Python Server ──> Rerun Visualizer
    (WebXR)                    (aiohttp)         (3D Display)
                                   │
                                   └──> REST API / Custom Apps
```

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

## Quick Start

```bash
# Generate SSL certificate (required for WebXR)
uv run python backend/generate_cert.py

# Start the server
uv run python backend/server.py

# On Quest 3 browser: https://<your-ip>:8000
# Accept SSL warning, click "Enter VR"
```

## UMI Dataset Recording

Record controller poses + camera in LeRobot format for robot learning.

### Recording

```bash
# List available cameras
uv run umi-recorder --list-cameras

# Using config file (recommended)
cp umi/config.example.yaml config.yaml  # Edit with your settings
uv run umi-recorder --config config.yaml

# Or with CLI arguments
uv run umi-recorder --repo-id user/dataset --camera 0:wrist:640x480

# With robot URDF (IK visualization)
uv run umi-recorder --config config.yaml --urdf /path/to/robot.urdf
```

**Config file** (`config.yaml`):
```yaml
repo_id: user/my_dataset
camera: 0:wrist:640x480
hand: right
fps: 30
urdf: /path/to/robot.urdf  # optional
tasks:
  - Pick up the cup
  - Place on shelf
```

**Controls:** `1-9` select task, `s` start, `e` end episode, `q` quit

### Playback

```bash
uv run umi-visualize --dataset ./datasets/my_dataset --list
uv run umi-visualize --dataset ./datasets/my_dataset --episode 0
```

### Gripper Calibration

```bash
uv run umi-calibrate-gripper
```

### Dataset Format

```
datasets/my_dataset/
├── meta/info.json, episodes.jsonl
├── data/episode_*.parquet
└── videos/observation.images.*/episode_*.mp4
```

## Python Client Library

Access controller tracking data programmatically.

### Installation

```bash
cd client && pip install -e .
```

### Basic Usage

```python
from quest_controller_client import QuestControllerClientSync

with QuestControllerClientSync('https://localhost:8000') as client:
    # Get latest pose
    pose = client.get_latest_pose()
    if pose:
        if pose.left:
            print(f"Left: {pose.left.position}")
        if pose.right:
            print(f"Right: {pose.right.position}")
```

### Streaming Real-Time Data

```python
from quest_controller_client import QuestControllerClientSync

def on_pose(pose):
    if pose.left:
        print(f"Left: {pose.left.position}")

with QuestControllerClientSync('https://localhost:8000') as client:
    client.stream(on_pose, blocking=True)
```

### Polling at Fixed Rate

```python
from quest_controller_client import QuestControllerClientSync

def on_pose(pose):
    if pose and pose.left:
        print(f"Position: {pose.left.position}")

with QuestControllerClientSync('https://localhost:8000') as client:
    # Poll at 30 Hz for 10 seconds
    client.poll(on_pose, rate_hz=30, duration=10)
```

### Async API (High Performance)

```python
import asyncio
from quest_controller_client import QuestControllerClient

async def main():
    async with QuestControllerClient('https://localhost:8000') as client:
        status = await client.get_status()
        print(f"Server running at {status.current_frame_rate} Hz")

        pose = await client.get_latest_pose()
        if pose and pose.left:
            print(f"Left controller: {pose.left.position}")

asyncio.run(main())
```

### Button States

```python
with QuestControllerClientSync('https://localhost:8000') as client:
    pose = client.get_latest_pose()
    if pose and pose.left:
        # Check if trigger is pressed (button 0)
        if pose.left.is_button_pressed(0):
            print("Trigger pressed!")
        # Get grip value (button 1)
        grip_value = pose.left.get_button_value(1)
        print(f"Grip: {grip_value:.2f}")
```

### Data Models

**PoseData:**
- `timestamp: float` - Pose timestamp
- `left: Optional[ControllerState]` - Left controller
- `right: Optional[ControllerState]` - Right controller
- `latency: float` - Latency in milliseconds

**ControllerState:**
- `position: Tuple[float, float, float]` - Position (x, y, z) in meters
- `orientation: Tuple[float, float, float, float]` - Quaternion (x, y, z, w)
- `is_button_pressed(index: int) -> bool` - Check button state
- `get_button_value(index: int) -> float` - Get button value (0-1)

## REST API

```bash
curl https://localhost:8000/api/pose/latest
curl https://localhost:8000/api/status
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| WebXR not working | Use HTTPS (`https://`), accept SSL warning |
| Can't connect | Check firewall, verify same network |
| No controller data | Enter VR mode, grant permissions |
| Low FPS | Use 5GHz WiFi, reduce update rate |

## Requirements

- Meta Quest 3
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Same network for Quest and server
