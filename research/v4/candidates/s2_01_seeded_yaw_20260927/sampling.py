"""Common scoped initialization hook: original XY sampler, explicit WXYZ yaw."""
import numpy as np
NAMES=('milk_mug_1','pill_dish_1','ketchup_1','basket_1')
NOMINAL={'milk_mug_1':(.065,-.18),'pill_dish_1':(-.17,.02),
         'ketchup_1':(-.09454,-.18),'basket_1':(-.0144,.26)}
SALT=0x53323031
def angles_for_seed(seed):
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),SALT]))
    return dict(zip(NAMES,map(float,rng.uniform(-10.,10.,len(NAMES)))))
def make_hook(original,angles,records):
    def sample(sampler,*args,**kwargs):
        placements=original(sampler,*args,**kwargs)
        result=dict(placements)
        for obj in sampler.mujoco_objects:
            if obj.name not in angles:continue
            pos,oldquat,placed_obj=result[obj.name]
            if placed_obj is not obj:raise ValueError('sampler object identity mismatch')
            if not np.allclose(oldquat,[0.,0.,0.,1.],atol=1e-12,rtol=0):
                raise ValueError('unexpected noncanonical authored orientation')
            if not np.all(np.abs(np.asarray(pos[:2])-NOMINAL[obj.name])<=.020+1e-12):
                raise ValueError('original sampler actual center beyond declared20mm')
            yaw=np.pi+np.deg2rad(angles[obj.name])
            quat=np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
            result[obj.name]=(pos,quat,placed_obj)
            records.append({'name':obj.name,'pos':list(map(float,pos)),
                            'original_quat':np.asarray(oldquat).tolist(),
                            'returned_quat_wxyz':quat.tolist(),
                            'yaw_delta_deg':angles[obj.name]})
        return result
    return sample
