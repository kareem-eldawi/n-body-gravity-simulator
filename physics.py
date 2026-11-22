import numpy as np
from body import Body
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 6.6743e-11
softening = 1e9

def calc_force(body1, body2):
    """Attractive pairwise force with a softened inverse-square magnitude."""

    displacement = body2.position - body1.position # body1 towards body 2

    r = np.linalg.norm(displacement)

    if r == 0:
        return np.zeros_like(displacement)

    else:
        unit_vector = displacement/r

        F = G * (body1.mass * body2.mass)/(r**2 + softening**2)

    force_vector = F * unit_vector

    return force_vector

def kinetic_energy(body):

    speed = np.linalg.norm(body.velocity)

    ke = 0.5 * body.mass * speed**2

    return ke

def potential_energy(bodies):
    totalpe = 0
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            displacement = bodies[i].position - bodies[j].position
            totalpe += -G * bodies[i].mass * bodies[j].mass / np.linalg.norm(displacement)

    return totalpe

def total_energy(bodies):
    ke = 0
    for f in bodies:
        ke += kinetic_energy(f)

    totalenergy = ke + potential_energy(bodies)

    return totalenergy


def total_force(target_body, all_bodies):

    total_force = np.zeros_like(target_body.position)

    for f in all_bodies:

        if f is target_body:
            pass
        else:
            total_force += calc_force(target_body, f)

    return total_force

def step(bodies, dt):
    """Advance every body using Euler-Cromer integration."""

    for b in bodies:
        force = total_force(b, bodies)
        b.acceleration = force/b.mass

    for b in bodies:
        b.velocity += b.acceleration * dt
        b.position += b.velocity * dt

def update(frame):
    for b in bodies2:
        x = data[b.name][frame][0]
        y = data[b.name][frame][1]
        dots[b.name].set_offsets([[x, y]])

        trail_x = [pos[0] for pos in data[b.name][:frame+1]]
        trail_y = [pos[1] for pos in data[b.name][:frame+1]]
        trails[b.name].set_data(trail_x, trail_y)
    return list(dots.values()) + list(trails.values())

def step_verlet(bodies, dt):
    """Advance every body using velocity Verlet and stored accelerations."""
    old_acceleration = {}

    for b in bodies:
        old_acceleration[b.name] = b.acceleration
        b.position += b.velocity * dt + 0.5 * b.acceleration * dt**2

    for b in bodies:
        force = total_force(b, bodies)
        b.acceleration = force/b.mass

    for b in bodies:
        b.velocity += 0.5 * (old_acceleration[b.name] + b.acceleration) * dt



if __name__ == "__main__":
    colors = {"Sun": "orange", "Earth": "blue", "Mars": "red"}
    sizes = {"Sun": 200, "Earth": 60, "Mars": 40}

    dt = 86400
    data = {}

   # Run 1: Euler-Cromer
    sun1 = Body(1.989e30, [0, 0], [0, 0], "Sun")
    earth1 = Body(5.972e24, [0, 29780], [1.496e11, 0], "Earth")
    mars1 = Body(6.417e23, [0, 24070], [2.279e11, 0], "Mars")
    bodies1 = [sun1, earth1, mars1]

    energy_euler = []
    for i in range(365):
        step(bodies1, dt)
        energy_euler.append(total_energy(bodies1))

    # Run 2: Verlet — completely separate, fresh bodies
    sun2 = Body(1.989e30, [0, 0], [0, 0], "Sun")
    earth2 = Body(5.972e24, [0, 29780], [1.496e11, 0], "Earth")
    mars2 = Body(6.417e23, [0, 24070], [2.279e11, 0], "Mars")
    bodies2 = [sun2, earth2, mars2]

    for b in bodies2:
            data[b.name] = []

    energy_verlet = []
    for i in range(365):
        step_verlet(bodies2, dt)
        for f in bodies2:
                    data[f.name].append(f.position.copy())

        energy_verlet.append(total_energy(bodies2))


    fig, ax = plt.subplots()

    dots = {}
    trails = {}
    for b in bodies2:
        dots[b.name] = ax.scatter([], [], color=colors[b.name], s=sizes[b.name])
        trails[b.name], = ax.plot([], [], color=colors[b.name], linewidth=0.8, alpha=0.5)


    print(len(data["Earth"]))
    print(data["Earth"][0])
    print(data["Earth"][-1])

    ani = animation.FuncAnimation(fig, update, frames=365, interval=30)

    ax.set_xlim(-2.5e11, 2.5e11)
    ax.set_ylim(-2.5e11, 2.5e11)
    ax.set_aspect('equal')

    plt.show()

    plt.figure()
    plt.plot(range(365), energy_euler, label="Euler-Cromer")
    plt.plot(range(365), energy_verlet, label="Verlet")
    plt.xlabel("Day")
    plt.ylabel("Total Energy (J)")
    plt.title("Energy Conservation: Euler-Cromer vs Verlet")
    plt.legend()
    plt.show()
