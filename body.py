import numpy as np


class Body:
    """A point mass with position, velocity, and acceleration in SI units."""

    def __init__(self, mass, velocity, position, name):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.name = name
        self.velocity = np.array(velocity, dtype=float)
        self.acceleration = np.zeros_like(self.position)

    def __repr__(self):
        return (
            f"Body(name={self.name}, mass={self.mass}, position={self.position}, "
            f"velocity={self.velocity}, acceleration={self.acceleration})"
        )
