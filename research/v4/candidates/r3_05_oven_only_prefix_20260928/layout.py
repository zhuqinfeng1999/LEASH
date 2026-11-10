"""One analytical appliance translation; own source/board untouched."""
import json,math
from pathlib import Path
H=Path(__file__).resolve().parent
P=H.parent/'r3_02_cart_tail_top_copy_20260928/r3_05_proposal_board_clearance.json'
SOURCE_XY=(-.08,-.16);OLD_OVEN_XY=(.06,.17);OVEN_Y=0.;CLEAR=.005

def proposal():
 r=json.loads(P.read_text());hx,hy=r['board_half_envelope_xy'];sourceY=(SOURCE_XY[1]-.02-hy,SOURCE_XY[1]+.02+hy)
 intersections=[]
 for g in r['actual_device_intersections']:
  disjoint=g['device_Y'][1]+CLEAR<sourceY[0]or sourceY[1]+CLEAR<g['device_Y'][0]
  intersections.append(dict(g,source_board_Y_disjoint5mm=disjoint))
 controlling=min(g['device_Xfront']for g in intersections if not g['source_board_Y_disjoint5mm'])
 required=SOURCE_XY[0]+.02+hx+CLEAR-controlling
 ovenX=r['oven_layout_unchanged'][0]+required
 chassis=json.loads((H.parent/'r3_02_mouth_aligned_side_layout_20260928/design.json').read_text())['all_actual_collidable_device_geoms']
 final=[]
 for g in chassis:
  if g['geom']in('microwave_1_g3','microwave_1_g4','microwave_1_g5','microwave_1_g7'):
   lo,hi,yl,yh=g['world_xy_interval_over_original_jitter'];final.append({'geom':g['geom'],'XY':[lo+ovenX-.02,hi+ovenX-.02,yl,yh]})
 assert max(g['XY'][1]for g in final)<.85 and max(abs(v)for g in final for v in g['XY'][2:])<.4
 return {'source_nominal_XY_unchanged':list(SOURCE_XY),'board_nominal_XY_unchanged':list(SOURCE_XY),'old_oven_nominal_XY':list(OLD_OVEN_XY),'new_oven_nominal_XY':[ovenX,OVEN_Y],'new_oven_yaw_nominal_rad':-math.pi/2,'oven_translation':[ovenX-OLD_OVEN_XY[0],OVEN_Y-OLD_OVEN_XY[1],0.],'board_half_rotated_XY': [hx,hy],'full_source_board_Y_interval':sourceY,'panel_records':intersections,'controlling_actual_geom':'microwave_1_g8','additional_translation_from_prior_cart_m':required,'selection':'minimum +X separation from all actual full-jitter panel envelopes whose Y overlaps unchanged full-jitter board/source; no pose/grid/IK','source_board_table_margins_XY':[.45-abs(SOURCE_XY[0])-.02-hx,.45-abs(SOURCE_XY[1])-.02-hy],'chassis_bottom_full_jitter_XY':final,'support':'Existing actual table[-.45,.45] joins existing real floor-cart X[.45,.85],Y[-.40,.40],top.900. Original door remains hinge-supported, not a floor/collision exemption. No dynamic cart-load claim.','constructor_control_simulations':0,'full_hand_prefix_new_device_collision_verified':False}
LAYOUT=proposal()
OVEN_XY=tuple(LAYOUT['new_oven_nominal_XY']);TRANSLATION=tuple(LAYOUT['oven_translation'])
