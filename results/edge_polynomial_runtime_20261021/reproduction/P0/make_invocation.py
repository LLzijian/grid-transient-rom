"""Generate only one frozen engineering invocation, without executing it."""
import argparse,runpy
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('qa','probe','benchmark'));p.add_argument('--source-commit',required=True);p.add_argument('--qa-retry-approval',type=Path);a=p.parse_args()
 runpy.run_path(str(Path(__file__).with_name('launch_stage.py')))['generate'](a.phase,a.source_commit,a.qa_retry_approval)
if __name__=='__main__':main()

