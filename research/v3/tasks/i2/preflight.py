"""I2 zero-action native scene preflight."""
import argparse
import json
from pathlib import Path
from research.v3.tasks.i1.preflight import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    print(json.dumps(run('I2',args.output,args.seed)),flush=True)

if __name__=='__main__':main()
