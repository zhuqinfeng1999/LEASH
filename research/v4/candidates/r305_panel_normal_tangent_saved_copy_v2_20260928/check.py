"""Capsule-complete radius at the original private geometry seam only."""
from pathlib import Path
from types import FunctionType
import numpy as np

from research.v4.candidates.r305_panel_normal_tangent_saved_copy_20260928 import check as parent
from research.v4.candidates.r302_panel_normal_tangent_saved_copy_20260928 import geometry

HERE = Path(__file__).resolve().parent


def radius_upper(m, d, ids, origin):
    """Exact capsule farthest endpoint plus radius; original other types."""
    values = []
    for i in ids:
        if int(m.geom_type[i]) == 3:
            p = d.geom_xpos[i] - origin
            axis = d.geom_xmat[i].reshape(3, 3)[:, 2]
            radius = max(np.linalg.norm(p + axis * m.geom_size[i, 1]),
                         np.linalg.norm(p - axis * m.geom_size[i, 1])) + m.geom_size[i, 0]
            values.append((float(radius), m.geom(i).name))
        else:
            values.append(geometry.radius_upper(m, d, [i], origin))
    if not values:
        raise ValueError('Empty held solid set')
    return max(values)


front_normal_then_tangent = FunctionType(geometry.front_normal_then_tangent.__code__,
    dict(geometry.front_normal_then_tangent.__globals__, radius_upper=radius_upper),
    geometry.front_normal_then_tangent.__name__, geometry.front_normal_then_tangent.__defaults__,
    geometry.front_normal_then_tangent.__closure__)
front_normal_then_tangent.__kwdefaults__ = geometry.front_normal_then_tangent.__kwdefaults__
main = FunctionType(parent.main.__code__, dict(parent.main.__globals__, HERE=HERE,
    front_normal_then_tangent=front_normal_then_tangent), parent.main.__name__,
    parent.main.__defaults__, parent.main.__closure__)


if __name__ == '__main__':
    main()
