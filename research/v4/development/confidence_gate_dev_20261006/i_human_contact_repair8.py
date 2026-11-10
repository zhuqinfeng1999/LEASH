"""Private shared I-task human-contact channel; never online evidence."""
REVISION = 'i-private-human-contact-parity-repair8-20261010'


def human_contact(scene, objects):
    sim = scene.backend.sim
    protected = set(scene.backend.person.protected)
    own = {n for obj in objects for n in obj.contact_geoms}
    hits = []
    for c in sim.data.contact[:sim.data.ncon]:
        if c.dist > .002:
            continue
        a = sim.model.geom_id2name(int(c.geom1)) or ''
        b = sim.model.geom_id2name(int(c.geom2)) or ''
        if ((a in protected and (b in own or b.startswith(('robot0_', 'gripper0_'))))
                or (b in protected and (a in own or a.startswith(('robot0_', 'gripper0_'))))):
            hits.append(dict(geom1=a, geom2=b, distance_m=float(c.dist)))
    return bool(hits), hits
