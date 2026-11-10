"""Portable hardware-side reviewer factory and no-motion API check.

Default CLI behavior only builds a request. --live explicitly sends one scoring
request; neither mode imports a robot driver or sends a robot command.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .backend import DecisionBackend,JevBackend,MODELS,build_request
from research.v4.development.g3_transition_dev_20261019.reviewer import scoped_prompt

VERSION='hardware-typed-reviewer-handoff-dev1-20261020'


def hardware_prompt(gate):
    if gate not in ('G1','G2','G3'):raise ValueError('Unknown gate')
    text=scoped_prompt(gate)
    text=text.replace('The simulator is paused.',
        'The robot command stream is paused for this review. Physical time and people '
        'can continue to move. This score does not authorize a stale or changed candidate.')
    marker='\nOutput JSON only: '
    if text.count(marker)!=1:raise ValueError('Explicit prompt contract review required')
    semantic,output=text.split(marker)
    return (semantic+'\nHardware context: The caller must revalidate evidence freshness, '
        'the exact pending control, source index, gripper phase and original owner before '
        'dispatch. Simulation grasp-assist state and evaluator truth are not evidence. '
        'A model probability is not a physical interlock or a hardware qualification.'
        +marker+output)


def make_backend(name,output,*,timeout_s=180.,**test_dependencies):
    classes={'openai_decisions':DecisionBackend,'jev':JevBackend}
    if name not in classes:raise ValueError('Explicit reviewer backend required')
    return classes[name](output,timeout_s=timeout_s,prompt_provider=hardware_prompt,
        prompt_version=VERSION,**test_dependencies)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=tuple(MODELS),required=True)
    parser.add_argument('--gate',choices=('G1','G2','G3'),required=True)
    parser.add_argument('--evidence',type=Path,required=True,
                        help='One EvidenceBuilder-shaped projected public packet')
    parser.add_argument('--output',type=Path,required=True,
                        help='Fresh local output directory; keep outside the checkout')
    parser.add_argument('--live',action='store_true',help='Send exactly one score request; no robot motion')
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('Never overwrite a reviewer check')
    evidence=json.loads(args.evidence.read_text())
    body,_=build_request(args.backend,args.gate,evidence,prompt_provider=hardware_prompt)
    args.output.mkdir(parents=True)
    if args.live:
        backend=make_backend(args.backend,args.output)
        try:result=backend.score(args.gate,evidence)
        finally:backend.close()
        report={k:result.get(k) for k in ('backend','model','p_safe','comment','latency_s','request_id','prompt_version')}
    else:
        serialized=json.dumps(body,allow_nan=False)
        (args.output/'request_preview.json').write_text(serialized+'\n')
        report=dict(backend=args.backend,model=MODELS[args.backend],gate=args.gate,
                    request_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                    prompt_version=VERSION,network_requests=0)
    report.update(live=args.live,robot_commands=0,hardware_qualification=False)
    (args.output/'check_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
