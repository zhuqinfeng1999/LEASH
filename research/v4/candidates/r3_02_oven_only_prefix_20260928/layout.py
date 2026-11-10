"""Actual R302 can and full 3D appliance envelopes; no Scene/physics/angle grid."""
import hashlib,itertools,math,json
from pathlib import Path
from xml.etree import ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r3_02_mouth_aligned_side_layout_20260928.design import extreme
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
OLD_OVEN_XY=(-.04,.17);SOURCE_XY=(-.08,-.16);CLEAR=.005
XML=ROOT/'research/v4/candidates/r3_02_cc0_soup_config_20260926/native_qa_seed0_gpu1/scene.xml'

def proposal():
 r=E.parse(XML).getroot();door=r.find(".//body[@name='microwave_1_microdoorroot']")
 dp=np.fromstring(door.get('pos'),sep=' ');D=Rotation.from_euler('z',-2.08).as_matrix();delta=math.radians(10)
 # Full visual/collision can envelope, including rims/tab, original vertical pose.
 source=[-.08-.02-.0365,-.08+.02+.0365,-.16-.02-.0365,-.16+.02+.0365,.900,1.015]
 rows=[]
 for g in r.findall('.//geom'):
  if not(g.get('name','').startswith('microwave_1_')and(int(g.get('contype','0'))or int(g.get('conaffinity','0')))):continue
  p=np.fromstring(g.get('pos','0 0 0'),sep=' ');sz=np.fromstring(g.get('size'),sep=' ');G=Rotation.from_euler('xyz',np.fromstring(g.get('euler','0 0 0'),sep=' ')).as_matrix()
  if g.get('type')=='box':points=[p+G@(sz*np.array(z))for z in itertools.product((-1,1),repeat=3)];rad=0.
  elif g.get('type')=='capsule':points=[p+G@np.array([0,0,z*sz[1]])for z in(-1,1)];rad=float(sz[0])
  else:raise ValueError('Unknown actual collidable oven shape')
  if g in list(door):points=[dp+D@u for u in points]
  bx=[];by=[];bz=[]
  for u,v,w in points:
   # Nominal yaw -90 + continuous original ±10°.
   lo,hi=extreme(u,v,-delta,delta);bx.extend([lo-.02-rad,hi+.02+rad])
   lo,hi=extreme(v,-u,-delta,delta);by.extend([lo-.02-rad,hi+.02+rad]);bz.extend([.9+w-rad,.9+w+rad])
  b=[min(bx),max(bx),min(by),max(by),min(bz),max(bz)]
  gy=max(b[2]-source[3],source[2]-b[3]);gz=max(b[4]-source[5],source[4]-b[5])
  rows.append({'geom':g.get('name'),'bounds_before_nominal_x_translation':b,'source_Y_gap_m':gy,'source_Z_gap_m':gz,'Y_or_Z_disjoint5mm':bool(max(gy,gz)>=CLEAR)})
 relevant=[g for g in rows if not g['Y_or_Z_disjoint5mm']]
 front=min(g['bounds_before_nominal_x_translation'][0]for g in relevant)
 X=source[1]+CLEAR-front
 for g in rows:
  b=g['bounds_before_nominal_x_translation'];g['final_world_xyz_interval']=[b[0]+X,b[1]+X,*b[2:]]
  g['source_X_gap_m']=b[0]+X-source[1];g['separated5mm_xyz']=bool(max(g['source_X_gap_m'],g['source_Y_gap_m'],g['source_Z_gap_m'])>=CLEAR-1e-12)
 bottoms=[g for g in rows if g['geom']in('microwave_1_g3','microwave_1_g4','microwave_1_g5','microwave_1_g7')]
 support=all(g['final_world_xyz_interval'][0]>=-.45 and g['final_world_xyz_interval'][1]<=.85 and g['final_world_xyz_interval'][2]>=-.40 and g['final_world_xyz_interval'][3]<=.40 for g in bottoms)
 return {'instance':'R3-02','source_nominal_XY':SOURCE_XY,'source_full_xyz_envelope':source,'old_oven_actual_nominal_XY':OLD_OVEN_XY,'new_oven_nominal_XY':[X,0.],'translation':[X-OLD_OVEN_XY[0],-.17,0.],'yaw_nominal_rad':-math.pi/2,'independent_source_device_xy_jitter_m':.02,'yaw_jitter_rad':delta,'original_open_hinge_rad':-2.08,'controlling_geom':min(relevant,key=lambda g:g['bounds_before_nominal_x_translation'][0])['geom'],'records':rows,'source_device_initial_separated5mm':all(g['separated5mm_xyz']for g in rows),'chassis_table_cart_supported_envelope':support,'actual_cart_XY':[[.45,.85],[-.40,.40]],'cart_top_m':.9,'old_r305_actual_nominal_not_inherited':[.23875321006107667,0.],'selection':'Unique smallest +X oven translation from actual own can full XYZ + all continuous actual device collision envelopes; Y/Z disjoint panels do not constrain X. No source/robot change. Sufficient conservative initial envelope, not swept robot proof.','old_native_XML_sha256':hashlib.sha256(XML.read_bytes()).hexdigest(),'Scene_controls_physics_GPU_models':0}
LAYOUT=proposal();TRANSLATION=tuple(LAYOUT['translation'])
if not(LAYOUT['source_device_initial_separated5mm']and LAYOUT['chassis_table_cart_supported_envelope']):raise ValueError('Own source/full-device/cart envelope gap')
if __name__=='__main__':
 (H/'layout.json').write_text(json.dumps(LAYOUT,indent=2)+'\n');print(json.dumps({k:LAYOUT[k]for k in('new_oven_nominal_XY','controlling_geom','source_device_initial_separated5mm','chassis_table_cart_supported_envelope')}))
