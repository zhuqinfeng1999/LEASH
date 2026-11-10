"""Render the existing P1 blade and bolster without changing their physics.

The accepted XML places these two meshes in collision group 0. Robosuite's
public cameras hide that group and the common loader does not duplicate its
meshes. Change only the compiled rendering group for those two geoms, leaving
the source XML, dimensions, mass, friction, collision flags and pose intact.
"""
from __future__ import annotations

from research.v4.candidates.p1_table_wrist_clearance_20260925.candidate import (
    P1TableWristClearanceFacade,
)
from research.v4.accelerated.e1_facade import E1TabletopFacade


VISUAL_SUFFIXES = ("blade", "bolster")
VISUAL_REVISION = "p1-render-only-blade-bolster-v1"


def reveal_existing_blade_and_bolster(sim, geom_names):
    """Select two existing loaded geoms by suffix; change render groups only.

    A caller supplies the object's existing contact geom names. The helper
    is reusable with any loaded robosuite environment but is applied only to
    P1 in this candidate. It never enables the global collision-render group.
    """
    model = sim.model._model
    changes = []
    for suffix in VISUAL_SUFFIXES:
        names = [name for name in geom_names if name.endswith("_" + suffix)]
        if len(names) != 1:
            raise ValueError(f"expected exactly one {suffix} geom, got {names}")
        gid = model.geom(names[0]).id
        old = int(model.geom_group[gid])
        if old != 0:
            raise ValueError(f"{suffix} collision group changed before visual override")
        changes.append({"geom": names[0], "before_group": old, "after_group": 1})
    # Validate both names/groups before mutating either one.
    for row in changes:
        model.geom_group[model.geom(row["geom"]).id] = row["after_group"]
    return tuple(changes)


class P1VisibleBladeFacade(P1TableWristClearanceFacade):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            sim = self.backend.sim
            item = self.backend.objects_dict[self.item_name]
            contact_names = tuple(item.contact_geoms)
            self.visual_render_changes = reveal_existing_blade_and_bolster(sim, contact_names)
            self.visual_revision = VISUAL_REVISION
            if self.backend.render_collision_mesh or not self.backend.render_visual_mesh:
                raise ValueError("P1 visibility candidate must retain normal public render groups")
            # Refresh the same native cameras after the render-only change.
            # No controller command or simulator time advance occurs here.
            if self.backend.use_camera_obs:
                self.raw = self.backend._get_observations(force_update=True)
        except BaseException:
            self.close()
            raise


class E1VisibleBladeQAOnlyFacade(E1TabletopFacade):
    """Use the same render-only helper for the authorized E1 native QA only.

    This is not installed in E1's live runner or shared task registration.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            item = self.backend.objects_dict[self.item_name]
            self.visual_render_changes = reveal_existing_blade_and_bolster(
                self.backend.sim, tuple(item.contact_geoms))
            self.visual_revision = VISUAL_REVISION
            if self.backend.render_collision_mesh or not self.backend.render_visual_mesh:
                raise ValueError("E1 QA must retain normal public render groups")
            if self.backend.use_camera_obs:
                self.raw = self.backend._get_observations(force_update=True)
        except BaseException:
            self.close()
            raise
