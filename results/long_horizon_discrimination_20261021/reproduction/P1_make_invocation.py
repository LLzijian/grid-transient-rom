"""Generate a single fixed long-horizon P1 evaluation; never execute a job."""
import argparse,runpy
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('evaluate',));p.add_argument('--seed',type=int,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
    h=runpy.run_path(str(Path(__file__).with_name('launch_stage.py')))
    h['generate'](a.phase,a.seed,a.source_commit)
if __name__=='__main__':main()


