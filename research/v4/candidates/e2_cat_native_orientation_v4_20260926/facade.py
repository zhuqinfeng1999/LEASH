"""Minimal visual conversion fix: preserve geometry/UV, correct body winding.

Original glTF materials are double-sided. The skinned body has inward final
winding, but its MuJoCo conversion rendered only front faces, exposing the
inside abdomen and distant eyes through culled back/head skin.
"""
from pathlib import Path

import numpy as np

from research.v4.candidates.e2_cat_face_v3_20260926.facade import FaceCandidateScene

HERE = Path(__file__).resolve().parent


class OrientationScene(FaceCandidateScene):
    def __init__(self, seed=0, *, camera_observations=False,
                 render_gpu_device_id=-1, orientation='A'):
        if orientation not in ('A', 'B'):
            raise ValueError(orientation)
        self.orientation = orientation
        self._visual_conversion_ready = False
        super().__init__(seed, candidate='C1',
                         camera_observations=camera_observations,
                         render_gpu_device_id=render_gpu_device_id)
        m = self.backend.sim.model
        mid = m.mesh_name2id('e2_cat_mesh_1')
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        # Triangles, UV indices and normal indices remain the same associations.
        # Reverse the inward body triangles only; eye mesh remains untouched.
        m.mesh_face[fa:fa+fn] = m.mesh_face[fa:fa+fn, ::-1].copy()
        m.mesh_facetexcoord[fa:fa+fn] = m.mesh_facetexcoord[fa:fa+fn, ::-1].copy()
        m.mesh_facenormal[fa:fa+fn] = m.mesh_facenormal[fa:fa+fn, ::-1].copy()
        self.frames = {key:self.frames[key] for key in self.frames.files}
        self.frames['faces_1'] = self.frames['faces_1'][:, ::-1].copy()
        self._visual_conversion_ready = True
        self._advance_pet(0.)
        if camera_observations:
            self.raw = self.backend._get_observations(force_update=True)

    def _advance_pet(self, elapsed):
        super()._advance_pet(elapsed)
        if not self._visual_conversion_ready or self.orientation == 'A':
            return
        # Exact same skin/rig pose/root/camera, whole cat rotated 180 degrees.
        # This is a static diagnostic contrast, not a backwards-moving candidate.
        m, d = self.backend.sim.model, self.backend.sim.data
        q = d.mocap_quat[self.mocap_id].copy()
        d.mocap_quat[self.mocap_id] = [-q[3], -q[2], q[1], q[0]]
        self.backend.sim.forward()
        self.current_visual_parts = []
        for index, (mid, meshadr, count, affine) in enumerate(self.meshes):
            gid = m.geom_name2id(f'e2_cat_visual_{index}')
            rotation = np.asarray(d.geom_xmat[gid]).reshape(3, 3)
            self.current_visual_parts.append(
                m.mesh_vert[meshadr:meshadr+count] @ rotation.T+d.geom_xpos[gid])

    def step(self, action):
        if self.orientation == 'B':
            raise RuntimeError('B is only a static opposite-direction contrast, not an event candidate')
        return super().step(action)
