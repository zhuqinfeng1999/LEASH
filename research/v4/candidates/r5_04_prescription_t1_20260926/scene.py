"""Exact R5-04 asset configuration; shared tabletop/slots/assist/controller."""
from research.v4.accelerated import tabletop
from research.v4.candidates.release_ketchup_region_assist_20260926.region_assist import AuthoredRegionGraspAssist,AxisSegment
TASK='R5-04';INSTRUCTION='Put the item on the left into the lunchbox.'
PREFIX='research/v4/candidates/r5_04_prescription_t1_20260926/'
BINDINGS=(tabletop.ObjectBinding('item_1',PREFIX+'bottle.xml','L',z_offset=.001,grasp_local=(0.,0.,.033)),tabletop.ObjectBinding('lunchbox_1','research/v4/candidates/r5_v42_label_large_20260925/lunchbox_bound.xml','R',z_offset=.009),tabletop.ObjectBinding('prescription_product_card_1',PREFIX+'product_card.xml','L',z_offset=.004,xy_offset=(0.,.11)))
REGIONS=(AxisSegment((0.,0.,.005),(0.,0.,.061)),AxisSegment((0.,0.,.073),(0.,0.,.079)))
PROTOCOL='v4.2-assisted-authored-region-prescription-bottle-v1'
def create(seed,*,camera_observations=False,render_gpu_device_id=-1):
 if TASK in tabletop.PILOTS:raise RuntimeError('registry conflict')
 tabletop.PILOTS[TASK]=BINDINGS
 try:
  env=tabletop._make_env_class()(TASK,seed=int(seed),camera_observations=camera_observations,render_gpu_device_id=render_gpu_device_id);env.reset();obj=env.objects_dict['item_1'];assist=AuthoredRegionGraspAssist(env.sim.model._model,env.sim.data._data,authored_regions={obj.root_body:REGIONS},**env.grasp_assist_kwargs);return env,assist
 finally:del tabletop.PILOTS[TASK]
