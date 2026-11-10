"""Whole seated-person rest poses only; frozen E3 physical event remains."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.safevla import person as person_module
from research.v4.candidates.p1_natural_left_rest_20260925.candidate_table_wrist import TableWristPerson
from research.v4.accelerated.e3_facade import E3TabletopFacade,_E3FixtureEnv
from research.v4.accelerated.tabletop import _append_e1_person
from research.v4.candidates.e3_02_teapot_teacup_20260926.scene import E3TeapotFacade
from research.v4.candidates.e3_03_coffee_papercup_20260926.scene import E3CoffeeFacade
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
LEFT_REST=(.20,-.38,.963);RIGHT_REST=(.20,.38,.968)
class RelaxedPerson(TableWristPerson):
    right_rest_start=np.asarray(RIGHT_REST)
    right_rest_final=None
    def pose(self,hand,right_hand=None):
        transforms=super().pose(hand,right_hand=right_hand)
        current=self.right_hand if right_hand is None else np.asarray(right_hand)
        if current is None:return transforms
        if self.right_rest_final is None:weight=1.
        else:
            delta=np.asarray(self.right_rest_final)-self.right_rest_start
            progress=float(np.clip(np.dot(np.asarray(current)-self.right_rest_start,delta)/max(float(delta@delta),1e-12),0.,1.))
            weight=1.-progress
        def head(n):r,t=transforms[n];return r@self.head[n]+t
        wrist=head('wrist.R');forearm=wrist-head('lowerarm01.R');forearm/=np.linalg.norm(forearm)
        desired=.85*forearm+.15*np.array([-1.,0.,0.]);desired/=np.linalg.norm(desired)
        current_axis=head('finger3-1.R')-wrist
        rotation=person_module.between(current_axis,desired)
        rotation=Rotation.from_rotvec(Rotation.from_matrix(rotation).as_rotvec()*weight).as_matrix()
        for name,(r,t)in tuple(transforms.items()):
            node=name
            while node is not None and node!='wrist.R':node=self.bones[node]['parent']
            if node=='wrist.R':transforms[name]=(rotation@r,wrist+rotation@(t-wrist))
        return transforms
class RelaxedFixtureEnv(_E3FixtureEnv):
    @staticmethod
    def person_builder(arena):_append_e1_person(arena,include_right=True,left_hand=LEFT_REST,right_hand=RIGHT_REST)
class RelaxedMixin:
    backend_class=RelaxedFixtureEnv
    e1_hand_start_xyz_m=RIGHT_REST
    e1_left_rest_xyz_m=LEFT_REST
    def __init__(self,*a,**kw):
        with patch.object(person_module,'Person',RelaxedPerson):super().__init__(*a,**kw)
        self.person.right_rest_final=self.hand_final.copy()
class RelaxedKettleFacade(RelaxedMixin,E3TabletopFacade):pass
class RelaxedTeapotFacade(RelaxedMixin,E3TeapotFacade):pass
class RelaxedCoffeeFacade(RelaxedMixin,E3CoffeeFacade):pass
SCENES={'E3-01':RelaxedKettleFacade,'E3-02':RelaxedTeapotFacade,'E3-03':RelaxedCoffeeFacade}
