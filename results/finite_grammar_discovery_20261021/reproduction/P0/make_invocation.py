"""Generate a registered finite-grammar command; never launch a process."""
import argparse,runpy
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('freeze-pool','qa','cost'));p.add_argument('--source-commit',required=True);p.add_argument('--qa-retry-approval',type=Path);a=p.parse_args()
    definitions=runpy.run_path(str(Path(__file__).with_name('launch_stage.py')))
    definitions['generate'](a.phase,a.source_commit,a.qa_retry_approval)
if __name__=='__main__':main()

