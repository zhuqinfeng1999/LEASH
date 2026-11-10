"""Root-only finite existing script recorder, with original R3 save/namespace."""
import inspect
from . import scripts
from .scripts import HERE,ROOT
from .qualify import hashes
from research.v4.candidates.r4_05_live_rim_grasp_20260926 import native_motion as parent
from research.v4.candidates.r3_03_t1_derived_20260926 import native_qa as original
SOURCE=inspect.getsource(parent.run_capture).replace('R4-05_REAL_SCRIPTED_MOTION_OFFLINE_NOT_METHOD','R3-06_REAL_SCRIPTED_MOTION_OFFLINE_NOT_METHOD').replace("instance_id='R4-05'","instance_id='R3-06'").replace("instruction='Put the item on the left into the crib.'","instruction='Put the item on the left into the microwave and close the door to heat it up.'").replace("'T1R4MagneticBoxScene'","'R3TabletopFacade'")
SOURCE=SOURCE.replace("super().__init__(*args,**kwargs);self._capture_closed=False","super().__init__(*args,**kwargs);self.initial_item_z=float(self.backend.sim.data.get_joint_qpos(self.backend.objects_dict[self.item_name].joints[0])[2]);self._capture_closed=False")
SOURCE=SOURCE.replace("['initial','actual_attachment','closed_actual','lifted_2cm','before_release','final_after_script']","['initial','actual_attachment','closed_actual','lifted_2cm','final_after_script']")
namespace=dict(vars(parent));namespace.update(HERE=HERE,ROOT=ROOT,scripts=scripts,hashes=hashes,original=original);exec(compile(SOURCE,__file__,'exec'),namespace);run_capture=namespace['run_capture'];exec(compile(inspect.getsource(parent.main),__file__,'exec'),namespace);main=namespace['main']
if __name__=='__main__':raise SystemExit(main())
