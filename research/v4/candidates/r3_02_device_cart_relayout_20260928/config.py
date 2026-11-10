"""One paired +X layout, no measured-success-dependent choices."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent.parent/'r3_02_mouth_aligned_side_layout_20260928'
BASE=json.loads((P/'design.json').read_text())
SOURCE_X=-.150
DELTA_X=SOURCE_X-BASE['source_nominal_xy'][0]
XY=(SOURCE_X,BASE['source_nominal_xy'][1])
OFFSET=(XY[0]+.08,XY[1]+.24)
OVEN_OFFSET=(.10+DELTA_X,-.24)
# Adjacent unbranded floor-supported cart, flush with the far table edge.
# Oven remains mostly on the original table; this cart supports its genuine
# small far-side overhang over the full original layout perturbation.
CART_X=(.45,.85);CART_Y=(-.40,.40);CART_TOP=.9;CART_THICK=.035

def proposal():
    hx,hy=BASE['source_cradle_xy_half_extent_over_yaw']
    chassis=[g for g in BASE['all_actual_collidable_device_geoms']
             if g['geom'] in ['microwave_1_g3','microwave_1_g4','microwave_1_g5','microwave_1_g7']]
    intervals=[]
    for row in chassis:
        lo,hi,yl,yh=row['world_xy_interval_over_original_jitter']
        intervals.append({'geom':row['geom'],'translated_interval':[lo+DELTA_X,hi+DELTA_X,yl,yh]})
    ymax=max(abs(v)for row in intervals for v in row['translated_interval'][2:])
    xmax=max(row['translated_interval'][1]for row in intervals)
    assert xmax<CART_X[1] and ymax<CART_Y[1]
    assert abs(SOURCE_X)+.02+hx<.45 and abs(XY[1])+.02+hy<.45
    return {'passed':True,'selection':'Root requested a normal source nominalX-.15 instead of the old near-base-.309779048; minimal paired translation to that declared station. This is not a globally optimized arm-reach minimum.',
      'source_nominal_xy':list(XY),'oven_nominal_xy':[.02+DELTA_X,0.],
      'paired_translation_x_m':DELTA_X,'relative_source_device_geometry_unchanged':True,
      'chassis_bottom_footprints_full_xy20mm_yaw10deg':intervals,
      'chassis_overhang_beyond_original_table_m':max(0,xmax-.45),
      'cart_top_xy':[list(CART_X),list(CART_Y)],'cart_top_z':CART_TOP,'cart_thickness':CART_THICK,
      'cart_bottom_leg_z':0.,'cart_support':'Original table and contiguous cart top jointly cover chassis bottom footprints. Open hinged door remains hinge-supported; it is not supposed to rest on a cart.',
      'source_whole_cradle_table_margins_xy_m':[.45-abs(XY[0])-.02-hx,.45-abs(XY[1])-.02-hy],
      'relative_panel_clearance_5mm_proof':'All source/device centers translate equally inX. Original independentXY±20mm/yaw±10deg bounds and their disjoint/separated pair inequalities translate without change.',
      'unproved':['actual dynamic cart load','native appearance/visibility','robot side approach/entry reach'],
      'scene_ik_physics_gpu_models':0}
