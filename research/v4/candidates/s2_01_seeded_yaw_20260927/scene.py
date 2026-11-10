"""Future common initialization version; immutable182 physics/risk inherited."""
from unittest.mock import patch
import numpy as np
from research.v4.candidates.s2_01_ordered_sampler_actual_20260927.scene import Scene as Parent
from .sampling import NAMES,angles_for_seed,make_hook

class Scene(Parent):
    protocol='S201-CC0-ordered20mm-centers+seeded-WXYZ-pi-yaw10deg-v1'
    def __init__(self,seed=0,*args,**options):
        from libero.libero.envs.regions.base_region_sampler import MultiRegionRandomSampler
        original=MultiRegionRandomSampler.sample
        records=[];angles=angles_for_seed(seed)
        try:
            with patch.object(MultiRegionRandomSampler,'sample',make_hook(original,angles,records)):
                super().__init__(seed=seed,*args,**options)
        finally:
            if MultiRegionRandomSampler.sample is not original:
                raise RuntimeError('scoped sampler binding was not restored')
        poses={}
        for name in NAMES:
            obj=self.env.env.objects_dict[name]
            q=np.asarray(self.env.sim.data.get_joint_qpos(obj.joints[0]),float)
            expected=np.array([np.cos((np.pi+np.deg2rad(angles[name]))/2),0.,0.,
                               np.sin((np.pi+np.deg2rad(angles[name]))/2)])
            similarity=min(1.,abs(float(np.dot(q[3:]/np.linalg.norm(q[3:]),expected))))
            rotation_error=2*np.arccos(similarity)
            matches=[r for r in records if r['name']==name]
            if not matches:raise ValueError('actual target not sampled '+name)
            closest=min(matches,key=lambda r:np.linalg.norm(q[:2]-r['pos'][:2]))
            poses[name]={'actual_qpos':q.tolist(),'yaw_delta_deg':angles[name],
                         'expected_quat_wxyz':expected.tolist(),
                         'post40warmup_rotation_error_rad':float(rotation_error),
                         'nearest_actual_returned_center':closest['pos'],
                         'post40warmup_XY_drift_m':float(np.linalg.norm(q[:2]-closest['pos'][:2]))}
            # This checks initialization correspondence, not a new risk/rest rule.
            if rotation_error>.005:raise ValueError('actual reset yaw differs after warmup '+name)
        self.yaw_initialization={'protocol':self.protocol,'seed':int(seed),'angles':angles,
                                 'sampler_returns':records,'actual_initial_poses':poses,
                                 'binding_restored':True,'pill_initializer_unchanged':True}
