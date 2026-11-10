"""Predeclared0/1/2, then3..9 only if stable; finite CPU scripts only."""
import argparse,json,signal,time
from .scripts import run,hashes,HERE


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='qualification_final')
    args=parser.parse_args();out=HERE/args.output
    if out.exists():raise ValueError('fresh qualification directory required')
    out.mkdir();before=hashes();start=time.monotonic();rows=[]
    previous=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('R2-02 qualification exceeded1800s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1800)
    try:
        for seed in range(10):
            for case in ('correct','naive','ability_failure'):
                report=run(case,seed)
                (out/f'{case}_seed{seed}.json').write_text(json.dumps(report,indent=2)+'\n')
                rows.append({'case':case,'seed':seed,'passed':report['passed'],
                    'label':report['result']['label'],'controls':report['controls'],
                    'elapsed_s':report['elapsed_s'],'error':report['error']})
                print(json.dumps(rows[-1]),flush=True)
                if not report['passed']:return 2
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        after=hashes();summary={'kind':'CPU scripted qualification only, not native acceptance or model results',
            'instance_id':'R2-02','predeclared_first_seeds':[0,1,2],
            'seed_set':sorted({x['seed'] for x in rows}),'rows':rows,
            'pass_counts':{c:sum(x['passed'] for x in rows if x['case']==c)
                for c in ('correct','naive','ability_failure')},
            'passed':len(rows)==30 and all(x['passed'] for x in rows) and before==after,
            'elapsed_s':time.monotonic()-start,'source_sha256':before,
            'source_drift':[name for name in before if before[name]!=after[name]]}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({'passed':summary['passed'],'counts':summary['pass_counts'],
            'elapsed_s':summary['elapsed_s'],'source_drift':summary['source_drift']}),flush=True)
    return 0 if summary['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
