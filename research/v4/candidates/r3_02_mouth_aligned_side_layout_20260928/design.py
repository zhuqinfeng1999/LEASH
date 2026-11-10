"""Continuous XY envelopes, original source/device jitter; no scene/anglegrid."""
import hashlib,itertools,json,math
from pathlib import Path
from xml.etree import ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
def extreme(a,b,low,high):
    ts={low,high};station=math.atan2(a,b)
    for k in range(-2,3):
        t=station+k*math.pi
        if low<=t<=high:ts.add(t)
    vals=[a*math.sin(t)+b*math.cos(t)for t in ts]
    return min(vals),max(vals)
def main():
    if(H/'design.json').exists():raise ValueError('Preserve first continuous design')
    path=ROOT/'research/v4/candidates/r3_02_cc0_soup_config_20260926/native_qa_seed0_gpu1/scene.xml'
    r=E.parse(path).getroot();delta=math.radians(10);theta=-2.08
    door=r.find(".//body[@name='microwave_1_microdoorroot']");dp=np.array([float(x)for x in door.get('pos').split()]);D=Rotation.from_euler('z',theta).as_matrix()
    sx=.075*math.cos(delta)+.058*math.sin(delta);sy=.075*math.sin(delta)+.058*math.cos(delta)
    source_y=.072;source_y_range=[source_y-.02-sy,source_y+.02+sy];rows=[]
    for g in r.findall('.//geom'):
        if not(g.get('name','').startswith('microwave_1_')and(int(g.get('contype','0'))or int(g.get('conaffinity','0')))):continue
        p=np.array([float(x)for x in g.get('pos','0 0 0').split()]);size=np.array([float(x)for x in g.get('size').split()]);G=Rotation.from_euler('xyz',[float(x)for x in g.get('euler','0 0 0').split()]).as_matrix()
        if g.get('type')=='box':points=[p+G@(size*z)for z in itertools.product((-1,1),repeat=3)];rad=0
        elif g.get('type')=='capsule':points=[p+G@[0,0,z*size[1]]for z in(-1,1)];rad=size[0]
        else:raise ValueError('Unexpected collidable oven shape')
        if g in list(door):points=[dp+D@p for p in points]
        bx=[];by=[]
        for u,v,_ in points:
            lo,hi=extreme(u,v,-delta,delta);bx.extend([.02-.02+lo-rad,.02+.02+hi+rad])
            lo,hi=extreme(v,-u,-delta,delta);by.extend([-.02+lo-rad,.02+hi+rad])
        bounds=[min(bx),max(bx),min(by),max(by)]
        gap=max(bounds[2]-source_y_range[1],source_y_range[0]-bounds[3])
        rows.append({'geom':g.get('name'),'world_xy_interval_over_original_jitter':bounds,'source_y_disjoint5mm':bool(gap>=.005),'y_gap_m':gap})
    blockers=[row for row in rows if not row['source_y_disjoint5mm']]
    front=min(row['world_xy_interval_over_original_jitter'][0]for row in blockers)
    source_x=front-.005-.02-sx;table_low=-.45+.02+sx
    result={'passed':bool(table_low<=source_x),'source_nominal_xy':[source_x,source_y],'source_nominal_x_supported_lower_bound':table_low,'source_cradle_xy_half_extent_over_yaw':[sx,sy],'source_y_interval_over_original_jitter':source_y_range,'device_xy_jitter_m':.02,'source_xy_jitter_m':.02,'yaw_degrees':10,'initial_fully_open_door_rad':theta,'all_actual_collidable_device_geoms':rows,'requires_X_separation':[row['geom']for row in blockers],'controlling_front_bound_m':front,'full_table_support_margin_x_m':source_x-table_low,'device_original_xml_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'Scene_IK_controls_GPU_models':0,'criterion':'Ydisjoint>=5mm excludes a panel; otherwise sourceX uses actual continuous fullX envelope. Conservative sufficient2D separation, not a claim of exact closest shape distance.'}
    (H/'design.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in('passed','source_nominal_xy','requires_X_separation','full_table_support_margin_x_m')}))
if __name__=='__main__':main()
