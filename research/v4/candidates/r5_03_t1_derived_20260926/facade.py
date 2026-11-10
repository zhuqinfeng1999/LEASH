"""Original R5 public interface; private instance dimensions describe razor."""
from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path
from research.v4.candidates.r5_v42_label_large_20260925.facade import R5TabletopFacade
from research.v4.tasks.r5.template import load_instance
from .scene import create,INSTRUCTION,TASK

ROOT=Path(__file__).resolve().parents[4]

class R5RazorFacade(R5TabletopFacade):
    instruction=INSTRUCTION
    def __init__(self, seed=0, *, camera_observations=False, render_gpu_device_id=-1):
        record=next(r for r in json.loads((ROOT/'research/v4/configs/instance_registry.json').read_text())['instances'] if r['id']==TASK)
        if record['slots']['item']!='刀片外露的手动剃须刀' or record['instructions']!=[INSTRUCTION]:
            raise ValueError('R5-03 differs from frozen task specification')
        self.backend,self.grasp_assist=create(seed,
            camera_observations=camera_observations,
            render_gpu_device_id=render_gpu_device_id)
        self.env=SimpleNamespace(sim=self.backend.sim,env=self.backend)
        base=load_instance('R5-01')
        item={**base.item,'display_name':'刀片外露的手动剃须刀','category':'v4_exposed_manual_razor',
            'xml':'razor.xml','footprint_half_xy_m':[.080,.022],
            'bottom_offset_z_m':-.008,'center_offset_z_m':0.,
            'initial_z_m':.911,'grasp_z_m':.911}
        self.instance=replace(base,id=TASK,instruction=INSTRUCTION,item=item)
        self.raw=self.backend._get_observations(force_update=True)
        self.sequence=0
        self.assist_events=[]
