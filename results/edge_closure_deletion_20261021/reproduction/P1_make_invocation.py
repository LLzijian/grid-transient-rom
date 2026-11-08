"""Generate a single fixed empty-closure P1 invocation; never execute a job."""
import argparse,runpy
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('train','freeze','evaluate'));p.add_argument('--seed',type=int);p.add_argument('--source-commit',required=True);a=p.parse_args()
    h=runpy.run_path(str(Path(__file__).with_name('launch_stage.py')))
    h['generate'](a.phase,a.seed,a.source_commit)
if __name__=='__main__':main()

