import unittest

import numpy as np

from barnes_hut import build_tree, force_from_node, step_verlet_bh
from body import Body
from physics import calc_force, total_force


class ForceTests(unittest.TestCase):
    def test_pair_forces_are_equal_and_opposite(self):
        a = Body(2e24, [0, 0], [0, 0], "a")
        b = Body(3e24, [0, 0], [1e11, 0], "b")
        np.testing.assert_allclose(calc_force(a, b), -calc_force(b, a))
        self.assertGreater(calc_force(a, b)[0], 0)

    def test_unapproximated_tree_matches_direct_forces(self):
        rng = np.random.default_rng(42)
        bodies = [Body(rng.uniform(1e23, 1e25), [0, 0], rng.uniform(-1e11, 1e11, 2), str(i)) for i in range(20)]
        root = build_tree(bodies)
        for body in bodies:
            np.testing.assert_allclose(force_from_node(body, root, 0), total_force(body, bodies), rtol=1e-12, atol=1e-8)

    def test_empty_and_single_body_systems(self):
        self.assertIsNone(build_tree([]))
        step_verlet_bh([], 10)
        body = Body(1e24, [1, 2], [0, 0], "only")
        step_verlet_bh([body], 10)
        np.testing.assert_allclose(body.position, [10, 20])
        np.testing.assert_allclose(body.velocity, [1, 2])

    def test_coincident_bodies_and_later_subdivision(self):
        bodies = [Body(i * 1e24, [0, 0], [0, 0], str(i)) for i in range(1, 4)]
        bodies.append(Body(4e24, [0, 0], [1e11, 0], "other"))
        root = build_tree(bodies)
        self.assertEqual(root.total_mass, sum(b.mass for b in bodies))
        for body in bodies:
            np.testing.assert_allclose(force_from_node(body, root, 0), total_force(body, bodies), rtol=1e-12, atol=1e-8)
        step_verlet_bh(bodies, 10)
        self.assertTrue(all(np.all(np.isfinite(b.position)) for b in bodies))

    def test_mass_recalculation_is_idempotent(self):
        bodies = [Body(1e24, [0, 0], [-1e11, 0], "a"), Body(2e24, [0, 0], [1e11, 0], "b")]
        root = build_tree(bodies)
        root.compute_mass()
        self.assertEqual(root.total_mass, 3e24)
        np.testing.assert_allclose(root.centre_of_mass, [1e11 / 3, 0])


if __name__ == "__main__":
    unittest.main()
