"""Private authored-region variant; no change to shared point assist.

One object/one weld remains. Each candidate is a <=5mm sampling of a real
asset-authored axis segment. Each point independently meets the original
between-pads AND local15/15/20mm test; tolerance is never enlarged.
"""
from dataclasses import dataclass
import math
import numpy as np
from research.v4.accelerated.grasp_assist import GraspAssist

PROTOCOL='v4.2-assisted-authored-region-ketchup-v1'

@dataclass(frozen=True)
class AxisSegment:
    start:tuple[float,float,float]
    end:tuple[float,float,float]
    def points(self):
        a,b=np.asarray(self.start,float),np.asarray(self.end,float)
        if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(np.r_[a,b]).all():raise ValueError('invalid authored segment')
        n=max(1,math.ceil(float(np.linalg.norm(b-a))/.005))
        return np.linspace(a,b,n+1)

KETCHUP_REGIONS=(AxisSegment((0.,0.,.010),(0.,0.,.083)),AxisSegment((0.,0.,.112),(0.,0.,.129)))

class AuthoredRegionGraspAssist(GraspAssist):
    def __init__(self,*args,authored_regions,**kwargs):
        super().__init__(*args,**kwargs)
        self._authored_regions={};self._region_last_match={}
        for name,segments in authored_regions.items():
            entries=[e for e in self._entries if e[0].object_body==name]
            if len(entries)!=1:raise ValueError('author region requires one registered object/one weld')
            zone,body,site,_=entries[0]
            if tuple(zone.tolerance_xyz)!=(.015,.015,.020):raise ValueError('region candidate requires unchanged common tolerance')
            points=np.concatenate([s.points() for s in segments],axis=0)
            if not len(points):raise ValueError('empty author region')
            self._authored_regions[site]=(body,points)
    def _eligible(self,site_id,tolerance_xyz):
        entry=self._authored_regions.get(site_id)
        if entry is None:return super()._eligible(site_id,tolerance_xyz)
        body,local_points=entry
        left=np.asarray(self.data.site_xpos[self._left]);right=np.asarray(self.data.site_xpos[self._right])
        gapvector=right-left;gap=float(np.linalg.norm(gapvector))
        if not math.isfinite(gap) or gap<=1e-6:return False
        rotation=np.asarray(self.data.xmat[body]).reshape(3,3)
        world=np.asarray(self.data.xpos[body])+local_points@rotation.T
        along=(world-left)@(gapvector/gap)
        center=np.asarray(self.data.site_xpos[self._center]);frame=np.asarray(self.data.site_xmat[self._center]).reshape(3,3)
        delta=(world-center)@frame
        valid=(along>=0)&(along<=gap)&np.all(np.abs(delta)<=np.asarray(tolerance_xyz),axis=1)
        indices=np.flatnonzero(valid)
        if not len(indices):self._region_last_match.pop(site_id,None);return False
        # Private audit only; the selected point cannot alter binding pose.
        i=int(indices[np.argmin(np.linalg.norm(delta[indices],axis=1))])
        self._region_last_match[site_id]={'local_m':local_points[i].tolist(),'world_m':world[i].tolist(),
            'gripper_delta_m':delta[i].tolist(),'along_pad_m':float(along[i]),'gap_m':gap}
        return True
