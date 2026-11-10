"""Own tray full envelope, original source preserved; source-only continuous extrema."""
import hashlib,itertools,math,json
from pathlib import Path
from xml.etree import ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation
from research.v4.candidates.r3_02_mouth_aligned_side_layout_20260928.design import extreme
H=Path(__file__).resolve().parent;ROOT=H.parents[3];OLD_OVEN_XY=(-.04,.17);SOURCE_XY=(-.08,-.16);CLEAR=.005
XML=ROOT/'research/v4/candidates/r3_02_cc0_soup_config_20260926/native_qa_seed0_gpu1/scene.xml'
OWN=ROOT/'research/v4/candidates/r3_06_original_aluminum_tray_20260926/assets/aluminum_tray.xml'

def proposal():
 r=E.parse(XML).getroot();door=r.find(".//body[@name='microwave_1_microdoorroot']");dp=np.fromstring(door.get('pos'),sep=' ');D=Rotation.from_euler('z',-2.08).as_matrix();delta=math.radians(10)
 # Actual normal180x120x45mm tray envelope contains every rim, rib, floor/wall.
 # +/-yaw full XY extrema analytic, no angle grid. Independent sourceXY20mm.
 hx=extreme(.06,.09,-delta,delta)[1];hy=extreme(.09,.06,-delta,delta)[1]
 source=[-.08-.02-hx,-.08+.02+hx,-.16-.02-hy,-.16+.02+hy,.900,.945]
 rows=[]
 for g in r.findall('.//geom'):
  if not(g.get('name','').startswith('microwave_1_')and(int(g.get('contype','0'))or int(g.get('conaffinity','0')))):continue
  p=np.fromstring(g.get('pos','0 0 0'),sep=' ');sz=np.fromstring(g.get('size'),sep=' ');G=Rotation.from_euler('xyz',np.fromstring(g.get('euler','0 0 0'),sep=' ')).as_matrix()
  if g.get('type')=='box':points=[p+G@(sz*np.array(z))for z in itertools.product((-1,1),repeat=3)];rad=0.
  elif g.get('type')=='capsule':points=[p+G@np.array([0,0,z*sz[1]])for z in(-1,1)];rad=float(sz[0])
  else:raise ValueError('Unknown actual appliance collision shape')
  if g in list(door):points=[dp+D@u for u in points]
  bx=[];by=[];bz=[]
  for u,v,w in points:
   lo,hi=extreme(u,v,-delta,delta);bx.extend([lo-.02-rad,hi+.02+rad]);lo,hi=extreme(v,-u,-delta,delta);by.extend([lo-.02-rad,hi+.02+rad]);bz.extend([.9+w-rad,.9+w+rad])
  b=[min(bx),max(bx),min(by),max(by),min(bz),max(bz)];gy=max(b[2]-source[3],source[2]-b[3]);gz=max(b[4]-source[5],source[4]-b[5]);rows.append({'geom':g.get('name'),'bounds_before_nominal_X':b,'source_Y_gap_m':gy,'source_Z_gap_m':gz,'Y_Z_disjoint5mm':bool(max(gy,gz)>=CLEAR)})
 relevant=[g for g in rows if not g['Y_Z_disjoint5mm']];X=source[1]+CLEAR-min(g['bounds_before_nominal_X'][0]for g in relevant)
 for g in rows:
  b=g['bounds_before_nominal_X'];g['final_world_XYZ']=[b[0]+X,b[1]+X,*b[2:]];g['source_X_gap_m']=b[0]+X-source[1];g['separated5mm']=bool(max(g['source_X_gap_m'],g['source_Y_gap_m'],g['source_Z_gap_m'])>=CLEAR-1e-12)
 bottom=[g for g in rows if g['geom']in('microwave_1_g3','microwave_1_g4','microwave_1_g5','microwave_1_g7')];support=all(g['final_world_XYZ'][0]>=-.45 and g['final_world_XYZ'][1]<=.85 and g['final_world_XYZ'][2]>=-.4 and g['final_world_XYZ'][3]<=.4 for g in bottom)
 # Check actual own asset primitive envelopes fit claimed normal bounding box.
 for g in E.parse(OWN).getroot().findall('.//geom'):
  if g.get('type')=='box':
   p=np.fromstring(g.get('pos','0 0 0'),sep=' ');s=np.fromstring(g.get('size'),sep=' ');assert np.all(np.abs(p)+s<=np.array([.09,.06,.0225])+1e-12)
  elif g.get('type')=='capsule':
   ft=np.fromstring(g.get('fromto'),sep=' ').reshape(2,3);rad=float(g.get('size'));assert np.all(np.max(np.abs(ft),axis=0)+rad<=np.array([.09,.06,.0225])+1e-12)
  else:raise ValueError('Unknown own tray geometry')
 return {'instance':'R3-06','own_asset':str(OWN.relative_to(ROOT)),'own_asset_sha256':hashlib.sha256(OWN.read_bytes()).hexdigest(),'original_source_nominal_XYZ':[-.08,-.16,.9225],'original_source_rim_grasp_local':[0,-.057,.018],'actual_own_normal_dimensions_m':[.18,.12,.045],'mass_kg':.12,'original_source_full_jitter_XYZ':source,'continuous_xy_half_extent':[hx,hy],'old_actual_oven_nominal_XY':OLD_OVEN_XY,'new_oven_nominal_XY':[X,0.],'translation':[X+.04,-.17,0.],'yaw_nominal_rad':-math.pi/2,'height_m':.9,'controlling_geom':min(relevant,key=lambda g:g['bounds_before_nominal_X'][0])['geom'],'records':rows,'all_source_panels_separated5mm':all(g['separated5mm']for g in rows),'chassis_union_table_cart_supported':support,'cart_XY':[[.45,.85],[-.4,.4]],'cart_top_m':.9,'old_XML_sha256':hashlib.sha256(XML.read_bytes()).hexdigest(),'scene_controls_IK_GPU_models':0,'selection':'Unique smallest +X full-device translation from actual own tray fullXYZ and independent original±20mm/±10° envelopes. Own source/rim/PARK unchanged, no other-instance source proof inherited.','source_after_actual55turn_admitted':False,'robot_full_prefix_clearance_admitted':False}
LAYOUT=proposal();TRANSLATION=tuple(LAYOUT['translation'])
if not(LAYOUT['all_source_panels_separated5mm']and LAYOUT['chassis_union_table_cart_supported']):raise ValueError('Own full tray/device/cart source geometry gap')
if __name__=='__main__':
 (H/'layout.json').write_text(json.dumps(LAYOUT,indent=2)+'\n');print(json.dumps({k:LAYOUT[k]for k in('new_oven_nominal_XY','controlling_geom','all_source_panels_separated5mm','chassis_union_table_cart_supported')}))
