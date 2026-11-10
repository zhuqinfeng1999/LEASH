"""One anatomically based cylindrical side-support pose, not endpoint nudges."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from research.safevla import person as person_module
from research.v4.candidates.e3_relaxed_initial_person_20260926.scene import RelaxedPerson
from research.v4.candidates.e3_05_milk_babybottle_20260926 import scene as original
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];INSTRUCTION=original.INSTRUCTION
BOTTLE_OUTER_RADIUS_M=original.BOTTLE_RADIUS
MIDDLE_MCP_BODY_TARGET_M=np.array([.025,.040,.085])
def side_rotation(p):
    f=p.head['finger3-1.R']-p.head['wrist.R'];f/=np.linalg.norm(f)
    across=p.head['finger5-1.R']-p.head['finger2-1.R'];across-=f*np.dot(across,f);across/=np.linalg.norm(across)
    return np.column_stack(([-1.,0,0],[0,0,1.],[0,1.,0]))@np.column_stack((f,across,np.cross(f,across))).T
class CylindricalPerson(RelaxedPerson):
    wrap_center=None
    def pose(self,hand,right_hand=None):
        tr=super().pose(hand,right_hand=right_hand)
        current=self.right_hand if right_hand is None else np.asarray(right_hand)
        if current is None or self.right_rest_final is None or self.wrap_center is None:return tr
        delta=np.asarray(self.right_rest_final)-self.right_rest_start
        progress=float(np.clip(np.dot(np.asarray(current)-self.right_rest_start,delta)/max(float(delta@delta),1e-12),0.,1.))
        def at(n):r,t=tr[n];return r@self.head[n]+t
        def rotate_subtree(joint,rotation):
            pivot=at(joint)
            for n,(r,t)in tuple(tr.items()):
                node=n
                while node is not None and node!=joint:node=self.bones[node]['parent']
                if node==joint:tr[n]=(rotation@r,pivot+rotation@(t-pivot))
        # Physiological forearm/palm roll, not downward wrist flexion: palm
        # axis points forward, finger rows span bottle body height.
        correction=side_rotation(self)@tr['wrist.R'][0].T
        rotate_subtree('wrist.R',Rotation.from_rotvec(Rotation.from_matrix(correction).as_rotvec()*progress).as_matrix())
        # Four articulated fingers follow exterior circular chords. Bone
        # lengths are retained exactly; no digit translation or hidden body.
        for finger in range(2,6):
            for segment in range(1,4):
                joint=f'finger{finger}-{segment}.R';pivot=at(joint)
                r,t=tr[joint]
                endpoint=at(f'finger{finger}-{segment+1}.R')if segment<3 else r@self.tail[joint]+t
                vec=endpoint-pivot;length=float(np.linalg.norm(vec))
                radial=pivot[:2]-np.asarray(self.wrap_center)[:2];radius=float(np.linalg.norm(radial))
                if radius<=length/2:raise ValueError('Real finger cannot form the predeclared exterior chord')
                angle=float(np.arctan2(radial[1],radial[0]));turn=2*np.arcsin(length/(2*radius))
                target=np.array([self.wrap_center[0]+radius*np.cos(angle+turn),self.wrap_center[1]+radius*np.sin(angle+turn),pivot[2]])
                correction=person_module.between(vec,target-pivot)
                rotate_subtree(joint,Rotation.from_rotvec(Rotation.from_matrix(correction).as_rotvec()*progress).as_matrix())
        return tr
class Scene(original.Scene):
    protocol_version='e3-05-cylindrical-body-side-support-v42-v4'
    def __init__(self,*a,**kw):
        with patch.object(original,'BottlePerson',CylindricalPerson):super().__init__(*a,**kw)
    def _target_hand_pose(self):
        sim=self.backend.sim;obj=self.backend.objects_dict[self.target_name]
        center=np.asarray(sim.data.body_xpos[sim.model.body_name2id(obj.root_body)]).copy()
        self.person.wrap_center=center.copy()
        # Solve wrist from the actual authored middle MCP-to-wrist offset.
        # This changes the human's normal support pose, not bottle placement.
        wrist=center+MIDDLE_MCP_BODY_TARGET_M-side_rotation(self.person)@(self.person.head['finger3-1.R']-self.person.head['wrist.R'])
        return wrist-np.array([.10,0.,.016])
