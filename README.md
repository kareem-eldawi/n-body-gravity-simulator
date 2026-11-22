# N-Body Gravitational Simulator

A two-dimensional gravitational simulation written in Python, with an interactive Pygame view, direct pairwise force calculations, and a Barnes-Hut quadtree approximation.

## Features

- Toggle the Sun and eight planets, zoom, and change simulation speed.
- Add individual stationary bodies or clusters of 50 or 200 orbiting bodies.
- Pause, reset, clear spawned bodies, and view orbital trails.
- Compare Euler-Cromer and velocity-Verlet integration with a Matplotlib demonstration.
- Explore Barnes-Hut accuracy against direct force calculations and benchmark the two methods.

## Run locally

Use Python 3.10 or newer. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python game.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. The interactive view needs a desktop display.

| Control | Action |
| --- | --- |
| Space | Pause or resume |
| Up / Down | Increase or decrease time per frame |
| Planet buttons | Toggle bodies and restart the current selection |
| Zoom slider | Change viewing scale |
| Click simulation area | Add a stationary body |
| +50 / +200 | Add a cluster |
| Clear spawned / Reset | Remove added bodies or reset the simulation |

## Other entry points

```bash
python physics.py       # Orbit animation and energy comparison
python barnes_hut.py    # Accuracy comparison and timing benchmark
```

The full benchmark includes a 3,200-body direct calculation and may take a while. Timings depend on the machine and body distribution; the console's rough crossover estimate is not a guaranteed performance result.

## Project layout

| File | Purpose |
| --- | --- |
| `body.py` | Mass, position, velocity, and acceleration |
| `physics.py` | Direct forces, integration, energy diagnostics, and plotting demo |
| `barnes_hut.py` | Quadtree construction, aggregate forces, and approximate integration |
| `game.py` | Pygame rendering and controls |

## Model and limitations

Positions, velocities, masses, and timesteps use SI units. The force uses an inverse-square magnitude with a softening term. Barnes-Hut groups sufficiently distant cells according to `theta`; smaller values generally improve accuracy at extra computational cost, and `theta=0` disables aggregate approximations.

This is an educational visualisation rather than an astronomical prediction tool. Motion is two-dimensional, initial orbits are simplified, body sizes are exaggerated, and collisions do not merge bodies. Large timesteps can destabilise orbits. Accelerations start at zero on creation, so the first Verlet step has an initialisation error. The energy diagnostic uses an unsoftened potential and should not be treated as an exact conservation check for close encounters under the softened force.

## Checks

```bash
python -m unittest discover -s tests -v
```
