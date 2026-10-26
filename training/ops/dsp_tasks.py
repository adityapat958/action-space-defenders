"""Cfg subclasses for ops/dsp_train.py (imported lazily by gym after the Isaac app is up)."""
from isaaclab.utils import configclass

from action_space_kit.tasks.direct.tello_defend.tello_defend_env_cfg import TelloDefendLiveEnvCfg

_ns = {}
for _k in range(7):
    _ns[f"LiveS{_k}"] = configclass(type(f"LiveS{_k}", (TelloDefendLiveEnvCfg,), {"__annotations__": {"curriculum_stage": int}, "curriculum_stage": _k}))
globals().update(_ns)
