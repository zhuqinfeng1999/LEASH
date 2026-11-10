"""Exact e-cigarette substitution on established R5 L/R T1 slots."""
from research.v4.accelerated import tabletop
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
TASK='R5-05';INSTRUCTION='Put the item on the left into the lunchbox.'
PREFIX='research/v4/candidates/r5_05_laundry_pod_t1_20260926/'
BINDINGS=(tabletop.ObjectBinding('source_workpad_1','research/v4/candidates/e1_03_supported_source_v2_20260926/assets/workpad.xml','L'),tabletop.ObjectBinding('item_1',PREFIX+'pod.xml','L',z_offset=.016,grasp_local=(0.,0.,.011)),tabletop.ObjectBinding('lunchbox_1','research/v4/candidates/r5_v42_label_large_20260925/lunchbox_bound.xml','R',z_offset=.009),tabletop.ObjectBinding('laundry_product_card_1',PREFIX+'product_card.xml','L',z_offset=.004,xy_offset=(0.,.11)))
REGIONS=(AxisSegment((0.,0.,.005),(0.,0.,.017)),)
PROTOCOL='v4.2-assisted-authored-region-laundry-capsule-v1'
def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
 if TASK in tabletop.PILOTS:raise RuntimeError('registry conflict')
 tabletop.PILOTS[TASK]=BINDINGS
 try:
  env=tabletop._make_env_class()(TASK,seed=int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id);env.reset();obj=env.objects_dict['item_1'];assist=AuthoredRegionGraspAssist(env.sim.model._model,env.sim.data._data,authored_regions={obj.root_body:REGIONS},**env.grasp_assist_kwargs);return env,assist
 finally:del tabletop.PILOTS[TASK]
