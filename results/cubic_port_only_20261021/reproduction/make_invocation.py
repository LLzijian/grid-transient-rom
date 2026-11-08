"""Prepare one explicit native invocation for this fixed study; never launches it."""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
FRESH=ROOT/"private_data/closure_mechanism_20261019/P3/fresh_design_pair01"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:json.dump(d,f,indent=2);f.write("\n")
p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("category",choices=("qa","cost","train","evaluate"));p.add_argument("source_commit");p.add_argument("cap",type=float);p.add_argument("--output",type=Path,required=True);p.add_argument("--extra",nargs=argparse.REMAINDER,default=[]);a=p.parse_args()
if not a.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in a.name):raise ValueError("simple unique invocation name required")

expected_caps={'qa':400.,'cost':500.,'train':2800.,'evaluate':240.}
if a.cap!=expected_caps[a.category]:raise ValueError('exact category cap required')
if a.category=='train':
    def value(flag):
        if a.extra.count(flag)!=1:raise ValueError('unique role flag required')
        return a.extra[a.extra.index(flag)+1]
    if value('--seed') not in ('6101','6102','6103') or value('--variant')!='P' or value('--target-step')!='1024':raise ValueError('only three P1024 roles')
output=a.output.resolve()
if not output.is_relative_to(BASE):raise ValueError("output outside study")
if a.category=='train' and output!=BASE/'models'/f"{value('--seed')}_P":raise ValueError('exact P role output required')
if a.category=='evaluate':
    if a.extra.count('--model')!=1:raise ValueError('one exact P endpoint required')
    model=Path(a.extra[a.extra.index('--model')+1]).resolve()
    allowed={BASE/'models'/f'{seed}_P'/'checkpoints/step001024/model.json':BASE/'evaluation'/f'{seed}_P'/'step001024' for seed in (6101,6102,6103)}
    if model not in allowed or output!=allowed[model]:raise ValueError('exact P evaluation role/output required')
if a.category in ('qa','cost') and output!=BASE/('qa01' if a.category=='qa' else 'cost01'):raise ValueError('single fixed stage output required')
if output.exists():raise FileExistsError("exclusive new output required")
command=BASE/"commands"/(a.name+".json");prior=BASE/"prior"/(a.name+".json");binding=BASE/"bindings"/(a.name+".json")
args=([str(BASE/"qa_driver.py")] if a.category=="qa" else ["-m","experiments.cubic_port_only_20261021",a.category])
args += ["--output",str(output),"--max-seconds",str({"qa":340.,"cost":440.,"train":2740.,"evaluate":180.}[a.category]),"--source-commit",a.source_commit]
if a.category!="qa":args += ["--binding",str(binding)]
args += a.extra
write(command,{"executable":str(ROOT/".venv-gpu/Scripts/python.exe"),"arguments":args,"arguments_line":subprocess.list2cmdline(args)})
receipts=sorted(str(p.resolve()) for p in BASE.glob("native_*/execution.json"));write(prior,receipts)
old=json.loads((FRESH/"native_adapt_6101_full8/execution.json").read_text(encoding='utf-8'))["source_before"]
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
pending=[ROOT/n for n in ("experiments/cubic_port_only_20261021.py","experiments/cubic_port_only_runtime_20261021.py","tests/test_cubic_port_only_20261021.py","tests/test_cubic_port_only_runtime_20261021.py","tests/test_cubic_port_only_native_20261021.py")]
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
for f in [Path(__file__),BASE/"implementation_cost.json",BASE/"launch_stage.py",BASE/"qa_driver.py",command,prior,FRESH/"inputs/train_combined.json",ROOT/"experiments/native_process_receipt_20261020.ps1",ROOT/"reports/cubic_port_only_20261021_protocol.md",ROOT/"reports/astra6_after_fused_runtime_20261021.md"]:
    files[f.relative_to(ROOT).as_posix()]=sha(f)
manifest=ROOT/"results/spatial_extrema_20261020/frozen_evaluation_endpoints.json"
if sha(manifest)!="b8c525f9b87709528c74f1b64724aace4b653e4a6948765da8471599cc1d0af4":raise ValueError("source manifest changed")
files[manifest.relative_to(ROOT).as_posix()]=sha(manifest)
for entry in json.loads(manifest.read_text(encoding='utf-8'))["endpoints"]:
    if entry["variant"]!="O":continue
    for key in ("model_path","training_path","checkpoint_path"):
        f=Path(entry[key]);files[f.relative_to(ROOT).as_posix()]=sha(f)
for f in [ROOT/"results/quadratic_memory_edge_20261020/cohort_identity.json",ROOT/"reports/cubic_port_only_20261021_plan_review.md"]:
    files[f.relative_to(ROOT).as_posix()]=sha(f)
if a.category in ("cost","train","evaluate"):
    f=ROOT/"private_data/closure_mechanism_20261019/P2/quartic_memory01/prepare01/prepare.json";files[f.relative_to(ROOT).as_posix()]=sha(f)
    f=ROOT/"private_data/closure_mechanism_20261019/P2/quartic_memory01/prepare01/schedule.json";files[f.relative_to(ROOT).as_posix()]=sha(f)
if a.category=="evaluate":
    f=BASE/"frozen_evaluation_endpoints.json";files[f.relative_to(ROOT).as_posix()]=sha(f)
for flag in ("--schedule-path","--admission","--recovery-progress","--recovery-authorization","--model"):
    if flag in args:
        f=Path(args[args.index(flag)+1]).resolve()
        if not f.is_relative_to(ROOT) or not f.is_file():raise ValueError("missing explicit input "+flag)
        files[f.relative_to(ROOT).as_posix()]=sha(f)
if "--model" in args:
    parent=Path(args[args.index("--model")+1]).resolve().parent
    for n in ("checkpoint.json","training.pt"):
        f=parent/n;files[f.relative_to(ROOT).as_posix()]=sha(f)

for f in [ROOT/'results/quartic_memory_20261020/frozen_evaluation_endpoints.json',ROOT/'results/quartic_memory_20261020/fixed_train_metrics_step001024.json',ROOT/'reports/cubic_port_only_20261021_plan_review.md']:
    files[f.relative_to(ROOT).as_posix()]=sha(f)

for e in json.loads((ROOT/'results/quartic_memory_20261020/frozen_evaluation_endpoints.json').read_text(encoding='utf-8'))['endpoints']:
    for key in ('model_path','training_path','checkpoint_path'):
        f=Path(e[key]);files[f.relative_to(ROOT).as_posix()]=sha(f)
    for name in ('model.json','training.pt','checkpoint.json'):
        f=Path(e['model_path']).parent.parent/'step000000'/name;files[f.relative_to(ROOT).as_posix()]=sha(f)
for receipt in receipts:
    f=Path(receipt);files[f.relative_to(ROOT).as_posix()]=sha(f)
write(binding,{"files":dict(sorted(files.items())),"mixed_hdf5_hashed":False,"train_array_identity":"prepare schedule.events and source metadata; exact allowed train datasets only","prior_source_receipt":str(FRESH/"native_adapt_6101_full8/execution.json")})
for name in ('manifest.json','model_N.json','model_D.json','synthetic.npz','rng.pt'):
    f=ROOT/'private_data/closure_mechanism_20261019/P0/fused_runtime01/prepare01'/name
    files[f.relative_to(ROOT).as_posix()]=sha(f)
# Fixture files must be frozen in the actual source binding before publishing it.
binding_record=json.loads(binding.read_text(encoding='utf-8'))
binding_record['files']=dict(sorted(files.items()))
binding.write_text(json.dumps(binding_record,indent=2)+'\n',encoding='utf-8')
worker=output/"worker.json"
if "--worker-output" in args:worker=Path(args[args.index("--worker-output")+1])/"worker.json"
launch=[str(BASE/"launch_stage.py"),"--command",str(command),"--output",str(BASE/("native_"+a.name)),"--worker",str(worker),"--binding",str(binding),"--prior",str(prior),"--source-commit",a.source_commit,"--category",a.category,"--cap",str(a.cap)]
write(BASE/"commands"/(a.name+"_launcher.json"),{"executable":str(Path(__import__("sys").executable).resolve()),"arguments":launch,"arguments_line":subprocess.list2cmdline(launch)})
print(json.dumps({"name":a.name,"binding_files":len(files),"command":str(command),"binding":str(binding),"prior_count":len(receipts),"launch_arguments":launch}))
