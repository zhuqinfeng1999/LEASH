"""Resource isolation for two processes owned by the existing formal queue.

No scene, action, policy, gate, budget, seed or scoring semantics live here.
Development retains its exclusive global lock; formal slots share that lock
and each exclusively own one GPU. Services and MCP bind serially, then run
concurrently with episode-local clients and toolkits.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
from unittest.mock import patch

MODE='parallel_B0_Ours_dual_gpu'
REVISION='formal-dual-gpu-isolation-20261013'

def slots(plan):
    execution=plan.get('execution') or {}
    count=execution.get('concurrent_episodes',1)
    if type(count)is int and count==1:return (1,)
    if type(count)is int and count==4:
        from research.v4.bridge.distributed_gpu_resources import configuration
        if execution.get('gpus')!=[0,1,2,3]or configuration(plan)is None:
            raise ValueError('Qualified explicit four-slot execution required')
        return (0,1,2,3)
    request=plan.get('latest_resume_request') or {}
    if (type(count)is not int or count!=2 or execution.get('gpus')!=[0,1]
            or request.get('execution_mode')!=MODE or not request.get('authority')):
        raise ValueError('Two GPU slots require explicit current authorization')
    return (0,1)

def validate_native_slot(plan,registration,gpu):
    available=slots(plan)
    if type(gpu)is not int or gpu not in available:
        raise ValueError('GPU differs from the authorized formal slots')
    if registration.get('execution_slot',registration.get('physical_gpu',1))!=gpu:
        raise ValueError('Native GPU differs from its exact registration')
    return available

def choose_for_slot(plan,gpu,queue):
    if gpu not in slots(plan):raise ValueError('Unauthorized GPU slot')
    if len(slots(plan))==1:
        candidates=queue(plan)
        return candidates[0]if candidates else None
    # Each method retains its frozen relative cell/seed/model order. A vacant
    # method pool gives its GPU to the other method, without creating retries.
    preferred='Ours'if gpu in(0,2)else'B0'
    resource_queues=None
    if len(slots(plan))==4:
        from research.v4.bridge.distributed_gpu_resources import prefer_cells
        resource_queues={method:queue(plan,method=method)for method in('B0','Ours')}
        eligible={c['id']for c in prefer_cells(plan,gpu,[c for rows in resource_queues.values()for c in rows])}
    for method in (preferred,'B0'if preferred=='Ours'else'Ours'):
        candidates=(queue(plan,method=method)if resource_queues is None else
                    [c for c in resource_queues[method]if c['id']in eligible])
        if candidates:return candidates[0]
    return None

@contextmanager
def robot_resource(*,here,read_state,update_state,formal,gpu,available_slots,run_id):
    dual=formal and tuple(available_slots)in((0,1),(0,1,2,3))
    with (Path(here)/'robot.lock').open('a')as global_lock:
        fcntl.flock(global_lock,(fcntl.LOCK_SH if dual else fcntl.LOCK_EX)|fcntl.LOCK_NB)
        with (Path(here)/f'robot_gpu_{gpu}.lock').open('a')as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=read_state()
            if formal and(not state.get('formal_allowed')or state.get('stop_requested')or state.get('source_custody')):
                raise RuntimeError('Formal authorization is not active')
            pid=os.getpid();key=str(gpu)
            def publish(data):
                mapping=dict(data.get('active_robot_slots')or{})if dual else{}
                mapping[key]=dict(pid=pid,gpu=gpu,run_id=run_id,
                    started_utc=datetime.now(timezone.utc).isoformat())
                data['active_robot_slots']=mapping
                data['active_robot_pid']=next(iter(mapping.values()))['pid']if len(mapping)==1 else None
            update_state(publish)
            try:yield
            finally:
                def remove(data):
                    mapping=dict(data.get('active_robot_slots')or{})
                    if(mapping.get(key)or{}).get('pid')==pid:mapping.pop(key,None)
                    data['active_robot_slots']=mapping
                    data['active_robot_pid']=next(iter(mapping.values()))['pid']if len(mapping)==1 else None
                update_state(remove)

@contextmanager
def startup_lock(here):
    with (Path(here)/'formal_service_startup.lock').open('a')as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield

@contextmanager
def isolated_mcp_startup(here):
    """Keep the original toolkit/server, but rule out cross-slot port races."""
    from rpent.planner.utils import http_mcp_server as module
    cls=module.HttpMcpServer;original_init=cls.__init__;original_start=cls.start
    def init(server,toolkit,**kwargs):
        server._formal_auto_port=not kwargs.get('port')
        return original_init(server,toolkit,**kwargs)
    def start(server,**kwargs):
        with startup_lock(here):
            if server._thread is None and server._formal_auto_port:
                server._port=module._pick_free_port(server._host)
            try:
                url=original_start(server,**kwargs)
                if(not getattr(server._server,'started',False)or server._thread is None
                        or not server._thread.is_alive()):
                    raise RuntimeError('The episode-owned MCP server did not bind successfully')
                return url
            except BaseException:
                server.stop()
                raise
    with patch.object(cls,'__init__',init),patch.object(cls,'start',start):yield
