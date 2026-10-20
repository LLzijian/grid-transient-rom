"""Prepare one explicit native invocation for this fixed study; never launches it."""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
FRESH=ROOT/"private_data/closure_mechanism_20261019/P3/fresh_design_pair01"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:json.dump(d,f,indent=2);f.write("\n")
p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("category",choices=("qa","prepare","cost","train","evaluate"));p.add_argument("source_commit");p.add_argument("cap",type=float);p.add_argument("--output",type=Path,required=True);p.add_argument("--extra",nargs=argparse.REMAINDER,default=[]);a=p.parse_args()
if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in a.name):raise ValueError("simple unique invocation name required")
output=a.output.resolve()
if not output.is_relative_to(BASE):raise ValueError("output outside study")
command=BASE/"commands"/(a.name+".json");prior=BASE/"prior"/(a.name+".json");binding=BASE/"bindings"/(a.name+".json")
args=([str(BASE/"qa_driver.py")] if a.category=="qa" else ["-m","experiments.temporal_objective_20261020",a.category])
args += ["--output",str(output),"--max-seconds",str(a.cap-60),"--source-commit",a.source_commit]
if a.category!="qa":args += ["--binding",str(binding)]
args += a.extra
write(command,{"executable":str(ROOT/".venv-gpu/Scripts/python.exe"),"arguments":args,"arguments_line":subprocess.list2cmdline(args)})
receipts=sorted(str(p.resolve()) for p in BASE.glob("native_*/execution.json"));write(prior,receipts)
old=json.loads((FRESH/"native_adapt_6101_full8/execution.json").read_text())["source_before"]
files={}
for name,h in old.items():
    # Reuse exact allowed train/physical leaf identities, never read mixed HDF5.
    if name.startswith(("experiments/","src/","tests/","reports/","results/")):continue
    if "closure_mechanism_20261019/P2/" in name:continue
    if "closure_mechanism_20261019/P3/" in name and not name.startswith(str((FRESH/"campaign01/train").relative_to(ROOT)).replace("\\","/")+"/"):continue
    if "/val/" in name or name.endswith(("val_conditions.json",".h5",".hdf5")):continue
    if name.startswith(("configs/","private_data/")):
        if sha(ROOT/name)!=h:raise ValueError("existing train/physical asset changed: "+name)
        files[name]=h
pending=[ROOT/n for n in ("experiments/temporal_objective_20261020.py","experiments/temporal_objective_runtime_20261020.py","tests/test_temporal_objective_20261020.py","tests/test_temporal_objective_runtime_20261020.py","tests/test_temporal_native_envelope_20261020.py")]
seen=set()
while pending:
    f=pending.pop().resolve()
    if f in seen:continue
    seen.add(f);files[f.relative_to(ROOT).as_posix()]=sha(f)
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8-sig"))):
        if isinstance(node,ast.Import):names=[x.name for x in node.names]
        elif isinstance(node,ast.ImportFrom):names=[node.module or ""]
        else:continue
        for name in names:
            if name.startswith("experiments."):name=name[12:]
            for base in (ROOT/"experiments",ROOT/"src",ROOT/"tests"):
                g=base.joinpath(*name.split(".")).with_suffix(".py")
                if g.is_file():pending.append(g)
                parts=name.split(".")
                for j in range(1,len(parts)+1):
                    init=base.joinpath(*parts[:j])/"__init__.py"
                    if init.is_file():pending.append(init)
for f in [Path(__file__),BASE/"implementation_cost.json",BASE/"launch_stage.py",BASE/"qa_driver.py",command,prior,FRESH/"inputs/train_combined.json",ROOT/"experiments/native_process_receipt_20261020.ps1",ROOT/"reports/temporal_objective_20261020_protocol.md",ROOT/"reports/astra6_after_quadratic_memory_20261020.md"]:
    files[f.relative_to(ROOT).as_posix()]=sha(f)
for seed in (6101,6102,6103):
    for n in ("model.json","training.pt","checkpoint.json"):
        f=ROOT/"private_data/closure_mechanism_20261019/P2/quadratic_memory_edge01/models"/f"{seed}_B/checkpoints/step000512"/n
        files[f.relative_to(ROOT).as_posix()]=sha(f)
for f in [ROOT/"results/quadratic_memory_edge_20261020/cohort_identity.json",ROOT/"reports/temporal_objective_20261020_plan_review.md"]:
    files[f.relative_to(ROOT).as_posix()]=sha(f)
for flag in ("--schedule-path","--admission","--recovery-progress","--recovery-authorization","--model"):
    if flag in args:
        f=Path(args[args.index(flag)+1]).resolve()
        if not f.is_relative_to(ROOT) or not f.is_file():raise ValueError("missing explicit input "+flag)
        files[f.relative_to(ROOT).as_posix()]=sha(f)
if "--model" in args:
    parent=Path(args[args.index("--model")+1]).resolve().parent
    for n in ("checkpoint.json","training.pt"):
        f=parent/n;files[f.relative_to(ROOT).as_posix()]=sha(f)
write(binding,{"files":dict(sorted(files.items())),"mixed_hdf5_hashed":False,"train_array_identity":"prepare schedule.events and source metadata; exact allowed train datasets only","prior_source_receipt":str(FRESH/"native_adapt_6101_full8/execution.json")})
worker=output/"worker.json"
if "--worker-output" in args:worker=Path(args[args.index("--worker-output")+1])/"worker.json"
launch=[str(BASE/"launch_stage.py"),"--command",str(command),"--output",str(BASE/("native_"+a.name)),"--worker",str(worker),"--binding",str(binding),"--prior",str(prior),"--source-commit",a.source_commit,"--category",a.category,"--cap",str(a.cap)]
write(BASE/"commands"/(a.name+"_launcher.json"),{"executable":str(Path(__import__("sys").executable).resolve()),"arguments":launch,"arguments_line":subprocess.list2cmdline(launch)})
print(json.dumps({"name":a.name,"binding_files":len(files),"command":str(command),"binding":str(binding),"prior_count":len(receipts),"launch_arguments":launch}))
