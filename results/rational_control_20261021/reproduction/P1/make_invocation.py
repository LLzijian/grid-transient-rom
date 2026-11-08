"""Generate only the next fixed rational-control P1 invocation."""
import argparse,runpy
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','evaluate'));p.add_argument('--group',choices=('0_rational','0_tanh','1_rational','1_tanh'));p.add_argument('--source-commit',required=True);a=p.parse_args()
 runpy.run_path(str(Path(__file__).with_name('launch_stage.py')))['generate'](a.phase,a.group,a.source_commit)
if __name__=='__main__':main()

