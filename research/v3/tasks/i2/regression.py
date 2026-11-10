"""I2 finite native Panda-action regression wrapper."""
import argparse
import json
from pathlib import Path
from research.v3.tasks.i1.regression import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('correct','naive'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    print(json.dumps(run('I2',args.mode,args.output,args.seed)),flush=True)

if __name__=='__main__':main()
