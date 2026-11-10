"""R5-03: realistic manual razor, existing validated R5 L/R slots and lunchbox."""
from research.v4.accelerated import tabletop
from research.v4.accelerated.grasp_assist import GraspAssist

TASK = 'R5-03'
INSTRUCTION = 'Put the item on the left into the lunchbox.'
PREFIX = 'research/v4/candidates/r5_03_t1_derived_20260926/'
BINDINGS = (
    tabletop.ObjectBinding('item_1', PREFIX+'razor.xml', 'L',
                           z_offset=.011, grasp_local=(0.,0.,0.)),
    tabletop.ObjectBinding('lunchbox_1',
        'research/v4/candidates/r5_v42_label_large_20260925/lunchbox_bound.xml',
        'R',z_offset=.009),
    tabletop.ObjectBinding('razor_product_card_1', PREFIX+'product_card.xml',
        'L',z_offset=.004,xy_offset=(0.,.11)),
)

def create(seed, *, camera_observations=False, render_gpu_device_id=-1):
    if TASK in tabletop.PILOTS:raise RuntimeError('task registry conflict')
    tabletop.PILOTS[TASK]=BINDINGS
    try:
        env=tabletop._make_env_class()(TASK,seed=int(seed),
            camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id)
        env.reset()
        assist=GraspAssist(env.sim.model._model,env.sim.data._data,
                           **env.grasp_assist_kwargs)
        return env,assist
    finally:del tabletop.PILOTS[TASK]
