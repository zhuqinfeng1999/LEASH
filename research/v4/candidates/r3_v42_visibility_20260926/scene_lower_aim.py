"""Second and final bounded R3 visibility candidate: lower wrist aim."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import mujoco
import numpy as np

from research.v4.accelerated import tabletop
from research.v4.candidates.r3_v42_tabletop_20260925.scene import R3TabletopFacade

NEUTRAL_QPOS = (
    1.6444398463355994, -0.7958122980981985, -2.3511957839418836,
    -1.742929440083914, 0.34520106308783005, 1.809686930230634,
    -1.4173548229676816,
)
SEARCH_INDEX = 2
POT_DY_M = .08
OVEN_DX_M = .04
_lock = Lock()


class LowerAimR3Scene(R3TabletopFacade):
    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1):
        source = Path(__file__).resolve().parent / "cpu_lower_aim_search_seed0.json"
        expected = json.loads(source.read_text())["ranked"][SEARCH_INDEX]["joint_qpos_rad"]
        if not np.allclose(NEUTRAL_QPOS, expected, rtol=0, atol=1e-10):
            raise RuntimeError("R3 lower-aim pose differs from CPU source")
        with _lock:
            old = tabletop.PARK_QPOS
            tabletop.PARK_QPOS = NEUTRAL_QPOS
            try:
                super().__init__(seed=seed, camera_observations=camera_observations,
                                 render_gpu_device_id=render_gpu_device_id)
            finally:
                tabletop.PARK_QPOS = old
        sim = self.backend.sim
        model, data = sim.model._model, sim.data._data
        pot = model.joint("moka_pot_1_joint0").id
        oven = model.joint("microwave_1_joint0").id
        data.qpos[model.jnt_qposadr[pot] + 1] += POT_DY_M
        data.qpos[model.jnt_qposadr[oven]] += OVEN_DX_M
        fixture = model.equality("r3_microwave_fixture").id
        model.eq_data[fixture, 3] += OVEN_DX_M
        data.qvel[model.jnt_dofadr[pot]:model.jnt_dofadr[pot] + 6] = 0
        data.qvel[model.jnt_dofadr[oven]:model.jnt_dofadr[oven] + 6] = 0
        sim.forward()
        if camera_observations:
            self.raw = self.backend._get_observations(force_update=True)
