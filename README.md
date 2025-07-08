# SQuAT Plan: Smooth Quadrotor Agile Trajectory Planning
![Python](https://img.shields.io/badge/python-3.9+-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Platform](https://img.shields.io/badge/platform-WSL--Debian%20%7C%20ROS-lightgrey)

**SQuAT Plan** is a Python-based framework for agile trajectory planning of quadrotors navigating through complex environments. It integrates nonlinear optimization (via GEKKO), obstacle avoidance, and both 3D and ROS-based visualizations.

## Project Structure

```
squat-plan/
├── run.py                          # Unified entry point
├── pyproject.toml                  # Modern build system config
├── src/
│   └── squatplan/
│       ├── __init__.py
│       ├── main.py                 # Core simulation runner
│       ├── trajopt.py              # Trajectory optimization logic
│       ├── plotter.py              # Matplotlib-based plotting
│       ├── quaternion.py           # Quaternion math utils
│       ├── forester.py             # Obstacle generation
│       └── sphere_example_rviz.py  # ROS RViz marker publishing
├── squat.rviz                      # RViz display config
├── LICENSE
├── README.md
└── presentation.pdf                # MAE 271D presentation
```

## Features

- **Trajectory optimization** using GEKKO with full or simplified dynamics
- **Obstacle avoidance** using geometric constraints
- **Quaternion-based orientation modeling**
- **3D visualizations** via Matplotlib and RViz
- **Synthetic forest generation** for randomized path planning scenarios

## Getting Started

### Dependencies

Clone the repo and install in editable mode:

```bash
git clone https://github.com/aaronjohnsabu1999/squat-plan.git
cd squat-plan
python3 -m venv .venv
source venv/bin/activate
pip install -e .[dev]
```

Install ROS dependencies if on Linux or a WSL:

```bash
sudo apt install ros-${ROS_DISTRO}-rospy \
                 ros-${ROS_DISTRO}-geometry-msgs \
                 ros-${ROS_DISTRO}-visualization-msgs
```

### Run Simulation

```bash
python run.py
```

To launch RViz in parallel:

```bash
roscore
# Then in another terminal:
python src/squatplan/sphere_example_rviz.py
```

## Output Example

- **Trajectory** and **state evolution** plots (position, velocity, quaternion, thrust, moments)
- **3D environment** with spherical/cylindrical obstacles and path trajectory

## Project Context

> Developed as a final project for MAE 271D — *Control and Trajectory Planning for Autonomous Aerial Systems* at UCLA.

**Contributors:**

- Aaron John Sabu  
- Ryan Nemiroff  
- Brett T. Lopez *(Instructor)*

Contact: `{aaronjs, ryguyn, btlopez}@ucla.edu`

## License

MIT License © 2025  
University of California, Los Angeles