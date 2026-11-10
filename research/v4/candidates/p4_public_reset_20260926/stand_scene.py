"""P4 v3: expose only four existing stand pieces, retain v2 reset physics."""
from .clear_scene import P4ClearViewFacade

STAND_GEOMS = (
    'iron_stand_left_rail_collision',
    'iron_stand_right_rail_collision',
    'iron_stand_stop_25',
    'iron_stand_stop_226',
)
VISUAL_REVISION = 'p4-render-only-stand-four-pieces-v3'


def reveal_existing_stand(model):
    """Validate the full scoped set before changing any rendering group."""
    changes = []
    for name in STAND_GEOMS:
        gid = model.geom(name).id
        old = int(model.geom_group[gid])
        if old != 0:
            raise ValueError(f'{name}: expected existing hidden group 0, got {old}')
        changes.append({'geom': name, 'before_group': old, 'after_group': 1})
    for row in changes:
        model.geom_group[model.geom(row['geom']).id] = 1
    return tuple(changes)


class P4VisibleStandFacade(P4ClearViewFacade):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.visual_render_changes = reveal_existing_stand(self.backend.sim.model._model)
            self.visual_revision = VISUAL_REVISION
            if self.backend.render_collision_mesh or not self.backend.render_visual_mesh:
                raise ValueError('P4 v3 requires the unchanged normal public render groups')
            if self.backend.use_camera_obs:
                self.raw = self.backend._get_observations(force_update=True)
        except BaseException:
            self.close()
            raise
