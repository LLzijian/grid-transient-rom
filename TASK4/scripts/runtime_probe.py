"""Read-only runtime/dependency probe; no dataset payloads or package installs."""
import _bootstrap
import importlib, json, platform, sys
from task4.paths import write_json, ROOT

out = {'python':sys.executable,'version':sys.version,'platform':platform.platform(),'modules':{}}
for name in ['numpy','scipy','h5py','torch','psutil','pytest']:
    try:
        m=importlib.import_module(name)
        out['modules'][name]={'version':m.__version__,'path':m.__file__}
        if name=='torch':
            out['cuda_available']=m.cuda.is_available()
            if out['cuda_available']:
                out['gpu']=m.cuda.get_device_name(0)
                out['gpu_total_bytes']=m.cuda.get_device_properties(0).total_memory
    except ImportError as exc:out['modules'][name]={'error':str(exc)}
write_json(ROOT/('reports/T4_0001_runtime_'+str(len(out['modules']))+'_'+sys.prefix.replace('\\','_').replace(':','').replace('/','_')+'.json'),out)
print(json.dumps(out,ensure_ascii=False,indent=2))
