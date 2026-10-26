# Pure-torch tests for tello_defend/assignment.py (no Isaac). Run: python tests/test_defend_assignment.py [path/to/assignment.py]
import os
import importlib.util, torch, sys
spec = importlib.util.spec_from_file_location("assignment", sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "source/action_space_kit/action_space_kit/tasks/direct/tello_defend/assignment.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# 2v2: both defenders nearer to attacker 0, but d1 is only slightly farther from a1 -> must split
ttg = torch.tensor([[[1.0, 5.0], [1.5, 2.0]]])
alive = torch.ones(1, 2, dtype=torch.bool)
assert m.assign_targets(ttg, alive, "nearest").tolist() == [[0, 0]]
assert m.assign_targets(ttg, alive, "minmax_ttg").tolist() == [[0, 1]]
assert m.assign_targets(ttg, alive, "min_sum_ttg").tolist() == [[0, 1]]
# one attacker dead -> both on the live one
alive2 = torch.tensor([[True, False]])
assert m.assign_targets(ttg, alive2, "minmax_ttg").tolist() == [[0, 0]]
# hysteresis: previous [1,0] costs max(5,1.5)=5 vs best 2 -> switch at h=0.3, keep at h=10
prev = torch.tensor([[1, 0]])
assert m.assign_targets(ttg, alive, "minmax_ttg", prev, 0.3).tolist() == [[0, 1]]
assert m.assign_targets(ttg, alive, "minmax_ttg", prev, 10.0).tolist() == [[1, 0]]
# previous pointing at dead attacker always switches
assert m.assign_targets(ttg, alive2, "minmax_ttg", torch.tensor([[1, 1]]), 10.0).tolist() == [[0, 0]]
# slot order: assigned first, then by ttg, dead last
tgt = torch.tensor([[0, 1]])
o = m.slot_order(ttg, alive, tgt); assert o.tolist() == [[[0, 1], [1, 0]]], o
o = m.slot_order(torch.tensor([[[3.0, 1.0, 2.0]]]), torch.tensor([[True, True, False]]), torch.tensor([[0]]))
assert o.tolist() == [[[0, 1, 2]]], o
# 4v4 batch: assignment is a permutation when all alive
E = 512; t = torch.rand(E, 4, 4) * 10
a = m.assign_targets(t, torch.ones(E, 4, dtype=torch.bool), "minmax_ttg")
assert (a.sort(-1).values == torch.arange(4)).all()
# matches brute-force optimum of max cost
import itertools
for e in range(20):
    best = min(itertools.permutations(range(4)), key=lambda p: (max(t[e, d, p[d]] for d in range(4)) + 0.01 * sum(t[e, d, p[d]] for d in range(4))).item())
    assert list(best) == a[e].tolist()
# intercept: target at (2,0,0) moving +y 0.5 m/s, speed 1 -> |(2, .5t)| = t -> t = 2/sqrt(0.75)
tt, aim = m.intercept_time(torch.tensor([[2.0, 0, 0]]), torch.tensor([[0, 0.5, 0]]), 1.0)
assert abs(tt.item() - 2 / 0.75**0.5) < 1e-4, tt
# near-equal speeds: one pair has t ~ 1e5 s -> must still split, not double up
huge = torch.tensor([[[1.0, 2.0e5], [1.5, 3.0e5]]])
assert m.assign_targets(huge, alive, "minmax_ttg").tolist() in ([[0, 1]], [[1, 0]]), m.assign_targets(huge, alive, "minmax_ttg")
nan = torch.tensor([[[1.0, float("nan")], [float("inf"), 2.0]]])
assert m.assign_targets(nan, alive, "minmax_ttg").tolist() == [[0, 1]]
o = m.slot_order(torch.tensor([[[1.0, 3.0e5, 2.0]]]), torch.tensor([[True, True, False]]), torch.tensor([[0]]))
assert o.tolist() == [[[0, 1, 2]]], o  # huge-but-live before dead
print("all assignment tests pass")
