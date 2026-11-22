import numpy as np
from physics import G, softening, calc_force


class Node:
    """A quadtree cell holding a body or an aggregate of child cells."""

    def __init__(self, centre, width):
        self.centre = np.array(centre, dtype=float)
        self.width = width
        self.body = None
        self.overlapping_bodies = []
        self.children = None
        self.total_mass = 0.0
        self.centre_of_mass = np.zeros_like(self.centre, dtype=float)

    def subdivide(self):
        d = self.width / 4
        self.children = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                offset = np.array([sx * d, sy * d])
                self.children.append(Node(self.centre + offset, self.width / 2))

    def child_for(self, position):
        index = 0
        if position[0] >= self.centre[0]:
            index += 2
        if position[1] >= self.centre[1]:
            index += 1
        return self.children[index]

    def insert(self, body):
        if self.children is not None:
            self.child_for(body.position).insert(body)

        elif self.body is None:
            self.body = body

        elif np.array_equal(body.position, self.body.position):
            # Identical positions cannot be separated by further subdivision.
            self.overlapping_bodies.append(body)

        else:
            overlapping_bodies = self.overlapping_bodies
            self.overlapping_bodies = []
            old_body = self.body          # stash it before you lose it
            self.body = None              # this node is no longer a leaf
            self.subdivide()              # create the four children
            self.child_for(old_body.position).insert(old_body)
            self.child_for(body.position).insert(body)
            for overlapping in overlapping_bodies:
                self.child_for(overlapping.position).insert(overlapping)

    def compute_mass(self):
        self.total_mass = 0.0
        self.centre_of_mass = np.zeros_like(self.centre, dtype=float)
        if self.children is not None:
            weighted = np.zeros_like(self.centre)
            for child in self.children:
                child.compute_mass()
                self.total_mass += child.total_mass
                weighted += child.total_mass * child.centre_of_mass
            if self.total_mass > 0:
                self.centre_of_mass = weighted / self.total_mass

        elif self.body is not None:
            self.total_mass = self.body.mass + sum(b.mass for b in self.overlapping_bodies)
            self.centre_of_mass = self.body.position.copy()


def force_from_node(body, node, theta):
    if node is None or node.total_mass == 0:
        return np.zeros_like(body.position)

    if node.body is not None:
        total = np.zeros_like(body.position)
        for source in [node.body] + node.overlapping_bodies:
            if source is not body:
                total += calc_force(body, source)
        return total

    else:
        displacement = (node.centre_of_mass - body.position)
        d = np.linalg.norm(displacement)

        if d == 0:
            return np.zeros_like(body.position)

        elif node.width / d < theta:
            F = G * body.mass * node.total_mass / (d**2 + softening**2)
            return F * (displacement / d)
        else:
            total = np.zeros_like(body.position)
            for child in node.children:
                total += force_from_node(body, child, theta)
            return total


def build_tree(all_bodies):
    """Build one force-evaluation tree, including empty or coincident systems."""
    if not all_bodies:
        return None
    max_coord = max(np.max(np.abs(b.position)) for b in all_bodies)
    root = Node([0.0, 0.0], max(max_coord * 4, 1.0))
    for b in all_bodies:
        root.insert(b)
    root.compute_mass()
    return root

def total_force_bh(body, all_bodies, theta):
    root = build_tree(all_bodies)
    if root is None:
        return np.zeros_like(body.position)
    return force_from_node(body, root, theta)

def step_verlet_bh(bodies, dt, theta=0.5):
    """Advance a frame, sharing one Barnes-Hut tree across all bodies."""
    old_acceleration = {}
    for b in bodies:
        old_acceleration[b.name] = b.acceleration
        b.position += b.velocity * dt + 0.5 * b.acceleration * dt**2

    root = build_tree(bodies)
    for b in bodies:
        b.acceleration = force_from_node(b, root, theta) / b.mass

    for b in bodies:
        b.velocity += 0.5 * (old_acceleration[b.name] + b.acceleration) * dt

if __name__ == "__main__":
    import time
    from body import Body
    from physics import total_force, step_verlet

    THETA = 0.5
    rng = np.random.default_rng(42)

    def random_bodies(n):
        """n bodies scattered in a disc, at solar-system scale."""
        out = []
        for i in range(n):
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(5e10, 5e12)
            position = [dist * np.cos(angle), dist * np.sin(angle)]
            out.append(Body(rng.uniform(1e24, 1e28), [0.0, 0.0], position, f"B{i}"))
        return out

    # ── Accuracy: compare against the brute-force reference ───────────────────
    print("Accuracy vs brute force (200 bodies)")
    print(f"{'theta':>7} {'mean error':>12} {'max error':>11}")
    bodies = random_bodies(200)
    for theta in (0.0, 0.3, 0.5, 1.0):
        root = build_tree(bodies)
        errors = []
        for b in bodies:
            exact = total_force(b, bodies)
            approx = force_from_node(b, root, theta)
            magnitude = np.linalg.norm(exact)
            if magnitude > 0:
                errors.append(np.linalg.norm(approx - exact) / magnitude * 100)
        print(f"{theta:>7.1f} {np.mean(errors):>11.3f}% {np.max(errors):>10.3f}%")

    # ── Speed: one full integration step, both methods ────────────────────────
    print(f"\nOne Verlet step, theta = {THETA}")
    print(f"{'bodies':>7} {'brute force':>13} {'barnes-hut':>12} {'speedup':>9}")
    for n in (100, 200, 400, 800, 1600, 3200):
        brute_set = random_bodies(n)
        bh_set = random_bodies(n)

        start = time.perf_counter()
        step_verlet(brute_set, 3600)
        brute_time = time.perf_counter() - start

        start = time.perf_counter()
        step_verlet_bh(bh_set, 3600, THETA)
        bh_time = time.perf_counter() - start

        print(f"{n:>7} {brute_time*1000:>11.1f}ms {bh_time*1000:>10.1f}ms "
              f"{brute_time/bh_time:>8.2f}x")

    print("\nBelow ~150 bodies the tree-building overhead outweighs the pruning,")
    print("so brute force wins. The gap widens with n from there.")
