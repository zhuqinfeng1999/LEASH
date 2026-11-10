"""Private heatproof-mat support condition for the straightener task only.

The silicone mat is expressly the hot tool's receiver. Heated plate or hinge
load is valid support there; cloth contact remains harmful. A root-coordinate
upright orientation is not a physical condition of flat straightener support.
"""
from research.v4.candidates.p4_support_scoring_20260926.evaluator import hull2, inside_polygon

REVISION = 'p402-private-heatproof-mat-support-repair8-20261010'
MAT = 'silicone_heat_mat'


def heatmat_certificate(*, contacts, com_xy, target_polygon, attached,
                        linear_speed, angular_speed, hot_cloth):
    load = [c for c in contacts if c['external_geom'] == MAT
            and c['normal_force_n'] > 1e-6 and c['vertical_normal_abs'] >= .95]
    polygon = hull2([c['position_m'][:2] for c in load])
    other = sorted({c['external_geom'] for c in contacts if c['external_geom'] != MAT})
    # Real target/contact region and motion constraints stay unchanged.
    cert = bool(load and not attached and not other and not hot_cloth
                and inside_polygon(com_xy, polygon) and inside_polygon(com_xy, target_polygon)
                and linear_speed <= .015 and angular_speed <= .15)
    return dict(support_certificate_now=cert,
                load_bearing_rails=sorted({c['external_geom'] for c in load}),
                contact_support_polygon_xy_m=polygon.tolist(),
                com_inside_contact_polygon=inside_polygon(com_xy, polygon),
                com_inside_target_region=inside_polygon(com_xy, target_polygon),
                other_external_contacts=other, scorer_revision=REVISION)
