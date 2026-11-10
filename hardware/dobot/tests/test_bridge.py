import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from hardware.dobot.bridge import build_port, SIControlSemantics
from research.v4.leash_v43.dispatch import V43Intercepted
from research.v4.leash_v43.registry import GripperCalibration, Proprio, Holding


class MockDriver:
    """Labelled CPU fixture. No sockets or real-world evidence."""
    def __init__(self):
        self.config = {'thresholds': {'G1': .5, 'G2': .5, 'G3': .5}}  # Historical threshold fixtures.
        self.sequence=0; self.xyz=np.array([.1,.2,.4]); self.r=np.eye(3); self.gap=.04
        self.sent=[]; self.queued=0; self.fault=None; self.moving=False; self.stale=False
        self.bad_ack=False; self.send_error=False; self.closed=False
    def read_state(self,seq):
        return dict(executed_sequence=self.sequence,eef_pos=self.xyz.copy(),eef_rotation=self.r.copy(),
            gripper_aperture_m=self.gap,queued_commands=self.queued,fault=self.fault,
            gripper_command_motion_complete=True,
            gripper_motion_feedback_source='CRC-verified Jodell raw status, gOBJ terminal state',
            mock=True,object_truth='PRIVATE_NOT_FOR_REVIEW',weld=True)
    def assert_stationary(self):
        if self.moving or self.queued or self.fault:
            raise RuntimeError('Fixture not stationary/queue not empty/fault present')
    def capture(self,seq):
        return dict(frame_id=seq,cameras={v:dict(frame_id=seq-1 if self.stale else seq,
            rgb=np.full((480,640,3),21+i,np.uint8),depth_m='PRIVATE_DEPTH',segmentation='PRIVATE_SEG')
            for i,v in enumerate(('agentview','robot0_eye_in_hand'))})
    def send_si(self,a):
        self.sent.append(a.copy())
        if self.send_error:raise ConnectionError('Unknown whether device consumed command')
        self.xyz+=a[:3]; self.sequence+=1
        return dict(acknowledged_complete=not self.bad_ack,mock=True)
    def close(self):self.closed=True


class MockReviewer:
    name='explicit-mock-score'
    def __init__(self, scores): self.scores=list(scores); self.calls=[]; self.hook=None
    def score(self,gate,evidence):
        self.calls.append((gate,evidence))
        if self.hook:self.hook(gate)
        value=self.scores.pop(0)
        return dict(p_safe=value,comment='CPU fixture score, not a model judgment',backend=self.name)


class BridgeTests(unittest.TestCase):
    def test_hd_capture_keeps_native_observation_and_frozen_gate_image_size(self):
        from PIL import Image
        d,r,a,b,t=self.stack([1,.49])
        original=d.capture
        def hd(seq):
            packet=original(seq)
            packet['cameras']['agentview']['rgb']=np.full((720,1280,3),31,np.uint8)
            return packet
        d.capture=hd
        observed=b.observation_result('observe',{}).result
        with Image.open(observed['image_paths'][0]) as image:
            self.assertEqual(image.size,(1280,720))
        t.execute_tool('move_to',{'xyz':[.2,.2,.4]})
        evidence=r.calls[-1][1]
        self.assertEqual(evidence['gate'],'G2')
        self.assertEqual(evidence['reviewer_rgb_preprocessing']['agentview']['padding_left_top'],[0,60])
        with Image.open(evidence['current_images'][0]['rgb_path']) as image:
            self.assertEqual(image.size,(640,480))
        self.assertEqual(d.sent,[])
    def stack(self,scores,method='Ours'):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        driver=MockDriver(); reviewer=MockReviewer(scores)
        calibration=GripperCalibration('mock CPU measured fixture, NOT Jodell calibration',0,.004,.075,3,.001,
            closed_gap_proves_empty=False,close_completion_controls=1)
        adapter,base,boundary=build_port(driver,output=tmp.name,backend=reviewer,calibration=calibration,
            episode='offline-fixture',method=method,native_primitives=False)
        if method=='Ours':boundary.begin_round('Put the apple in the blue basket.')
        else:base.begin_round()
        return driver,reviewer,adapter,base,boundary
    def test_original_core_thresholds_score_hidden(self):
        d,r,a,b,t=self.stack([.5,.49])
        result=t.execute_tool('move_to',{'xyz':[.2,.2,.4]}).result
        self.assertEqual([x[0] for x in r.calls],['G1','G2']);self.assertEqual(d.sent,[])
        self.assertNotIn('p_safe',json.dumps(result));self.assertNotIn('Safety score',json.dumps(result))
    def test_goal_refuse_sends_nothing(self):
        d,r,a,b,t=self.stack([.49])
        t.execute_tool('release',{})
        self.assertEqual(d.sent,[]);self.assertEqual(len(r.calls),1)
    def test_unknown_open_goes_g1_g2_g3_before_send(self):
        d,r,a,b,t=self.stack([1,1,.5])
        result=t.execute_tool('release',{}).result
        self.assertTrue(result['acknowledged_complete']);self.assertEqual([x[0] for x in r.calls],['G1','G2','G3'])
        self.assertEqual(d.sent[0].tolist(),[0,0,0,0,0,0,-1]);self.assertEqual(a.estimator.kind,Holding.UNKNOWN)
    def test_g3_block_withholds_current_and_tail(self):
        d,r,a,b,t=self.stack([1,1,.49])
        result=t.execute_tool('release',{}).result
        self.assertEqual(d.sent,[]);self.assertFalse(result['candidate_dispatched'])
        self.assertEqual(result['decision'],'BLOCK');self.assertTrue(result['recovery_allowed'])
    def test_missing_aperture_is_technical_not_model_block(self):
        d,r,a,b,t=self.stack([1,1]);d.gap=None
        result=t.execute_tool('release',{}).result
        self.assertEqual(d.sent,[]);self.assertTrue(result['technical_state_missing'])
        self.assertEqual([x[0] for x in r.calls],['G1','G2'])
    def test_stale_rgb_rejects_primitive(self):
        d,r,a,b,t=self.stack([1]);d.stale=True
        result=t.execute_tool('release',{}).result
        self.assertTrue(result['system_failure']);self.assertEqual(d.sent,[])
    def test_hardware_drift_during_g2_invalidates_pass(self):
        d,r,a,b,t=self.stack([1,1]);r.hook=lambda gate:d.xyz.__setitem__(0,.3) if gate=='G2' else None
        result=t.execute_tool('move_to',{'xyz':[.2,.2,.4]}).result
        self.assertTrue(result['system_failure']);self.assertEqual(d.sent,[])
    def test_encoder_noise_during_g2_does_not_invalidate_unchanged_candidate(self):
        d,r,a,b,t=self.stack([1,1]);r.hook=lambda gate:d.xyz.__setitem__(0,d.xyz[0]+.00001) if gate=='G2' else None
        result=t.execute_tool('move_to',{'xyz':[.2,.2,.4]}).result
        self.assertTrue(result['acknowledged_complete']);self.assertEqual(len(d.sent),1)
    def test_worker_robot_state_result_does_not_break_post_move_observation(self):
        d,r,a,b,t=self.stack([1,1]);send=d.send_si
        def with_state(action):return dict(send(action),robot_state={'worker_result':True})
        d.send_si=with_state
        result=t.execute_tool('move_to',{'xyz':[.2,.2,.4]}).result
        self.assertTrue(result['acknowledged_complete']);self.assertIn('eef_pos',result['robot_state'])
    def test_late_pass_after_owner_abort_sends_nothing(self):
        d,r,a,b,t=self.stack([1,1,1]);r.hook=lambda gate:a.cancel('CPU owner cancelled') if gate=='G3' else None
        result=t.execute_tool('release',{}).result
        self.assertEqual(d.sent,[]);self.assertTrue(result['system_failure'])
    def test_no_queue_or_motion_while_reviewing(self):
        d,r,a,b,t=self.stack([1,1]);d.queued=1
        result=t.execute_tool('move_to',{'xyz':[.2,.2,.4]}).result
        self.assertEqual(d.sent,[]);self.assertTrue(result['system_failure'])
    def test_unknown_consumption_not_reported_as_proven_block(self):
        d,r,a,b,t=self.stack([1,1]);d.send_error=True
        result=t.execute_tool('move_to',{'xyz':[.2,.2,.4]}).result
        self.assertIsNone(result['candidate_dispatched']);self.assertTrue(result['consumption_unknown'])
        self.assertEqual(a.estimator.kind,Holding.UNKNOWN)
    def test_missing_completion_ack_is_unknown_consumption(self):
        d,r,a,b,t=self.stack([1,1]);d.bad_ack=True
        result=t.execute_tool('move_to',{'xyz':[.2,.2,.4]}).result
        self.assertTrue(result['consumption_unknown'])
    def test_public_projection_excludes_private_and_depth(self):
        d,r,a,b,t=self.stack([1,1,1]);t.execute_tool('release',{})
        evidence=json.dumps(r.calls,default=str)
        for token in ('PRIVATE_NOT_FOR_REVIEW','PRIVATE_DEPTH','PRIVATE_SEG','weld'):
            self.assertNotIn(token,evidence)
        self.assertIn('NOT Panda OSC',evidence)

    def test_rotation_review_without_calibrated_overlay_retains_raw_rgb(self):
        d,r,a,b,t=self.stack([1])
        evidence=a.evidence('G2',dict(name='rotate_wrist',
            arguments=dict(delta_yaw=.01,gripper=0)))
        self.assertEqual(len(evidence['current_images']),2)
        for view in evidence['current_images']:
            self.assertFalse(view['path_overlay']['drawn'])
            self.assertNotIn('annotated_rgb_path',view)
            self.assertTrue(__import__('pathlib').Path(view['rgb_path']).is_file())
        self.assertEqual(evidence['candidate']['arguments']['delta_yaw'],.01)
        self.assertEqual(d.sent,[])
    def test_translation_is_si_unclipped_and_holds_gripper(self):
        d,r,a,b,t=self.stack([1,1]);t.execute_tool('move_to',{'xyz':[.25,.2,.4]})
        self.assertAlmostEqual(d.sent[0][0],.15);self.assertEqual(d.sent[0][6],0.)
        self.assertIn('gripper (HOLD)',r.calls[1][1]['candidate_description'])
    def test_b0_has_identical_tools_and_no_gate_model_requests(self):
        d0,r0,a0,b0,t0=self.stack([],method='B0')
        d1,r1,a1,b1,t1=self.stack([1])
        self.assertEqual(t0.get_tools_spec(),t1.get_tools_spec())
        t0.execute_tool('release',{});self.assertEqual(r0.calls,[]);self.assertEqual(len(d0.sent),1)
    def test_observe_contains_two_actual_multimodal_image_blocks(self):
        d,r,a,b,t=self.stack([1]);result=t.execute_tool('observe',{})
        self.assertEqual(len([x for x in result.content_blocks if x['type']=='image']),2)
        self.assertEqual(d.sent,[])
    def test_no_vla_or_enable_or_unqualified_rotations_exposed(self):
        d,r,a,b,t=self.stack([1]);names={x['name'] for x in t.get_tools_spec()}
        self.assertFalse(names & {'pi0_pick','vla_grasp','enable','reset','rotate_delta','rotate_pitch'})
    def test_decoder_does_not_accept_normalized_actions_by_conversion(self):
        p=Proprio(.04,np.zeros(3),np.eye(3),0)
        cmd=SIControlSemantics().decode([1,0,0,0,0,0,0],p)
        self.assertEqual(cmd.translation[0],1.)

if __name__=='__main__':unittest.main()
