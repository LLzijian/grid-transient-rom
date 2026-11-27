"""Render the paired pendulum fit, noise and power receipts without new integration."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.osgkd_v2_refits.table3 import f, table
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io


RUN = 'osgkd_v2_wp3_pendulum_20261127'
PUBLIC = ROOT/'results'/RUN
LABELS = {'osgkd_full':'OSGKD 全量','osgkd_few':'OSGKD 少量','source_frozen':'冻结 Layer 1 编码器',
          'random_frozen':'冻结随机编码器','neural_full':'图 Neural ODE 全量','neural_few':'图 Neural ODE 少量',
          'closed_passive':'封闭被动摆阵','closed_conservative':'封闭保守摆阵'}


def render():
    reg = io.read(PUBLIC/'release.json')
    val = io.read(PUBLIC/'evaluate01/summary.json')
    nominal, noisy, physical = [], [], []
    for job in reg['jobs']:
        result = io.read(io.verified(val['terminal_jobs'][job['id']]))
        io.require(result['job']==job,'fixed pendulum job identity differs')
        name = LABELS[job['method']]
        prefix = [job['system_seed'],name,job['train_count'],result['fit']['accepted']]
        for key, rows in (('nominal',nominal),('noisy',noisy)):
            metrics = result.get(key)
            if metrics is None:
                rows.append([*prefix,'—','—','—','—','—','—',result['fit']['status']])
                continue
            coverage = f"{metrics['coverage']}/{metrics['raw_count']}"
            windows = metrics['windows']
            last = windows['400'] or {}
            m = last.get('rmse',{})
            risk = ' / '.join(f((windows[str(h)] or {}).get('risk_mean_tail'),5) for h in (100,200,400))
            rows.append([*prefix,coverage,risk,f(m.get('theta'))+' / '+f(m.get('omega')),
                         f(last.get('common_speed_rmse')),f(m.get('tau_m'))+' / '+f(m.get('tau_e')),
                         f(metrics['inference_per_event_s'],5),result['fit']['status']])
        if 'nominal' in result:
            power = result['power_check'] or {}
            physical.append([*prefix,result['parameters'],result['trainable_parameters'],result['evolved_state_dimension'],
                             f(power.get('chain_residual_abs_max'),14),f(power.get('reported_power_min')),
                             f(power.get('reported_power_max'))])
    cols = ['系统','方法','训练样本数','接受更新','覆盖','稳健 1/2/4 s 风险','4 s 相角/转速 RMSE',
            '4 s 共同转速 RMSE','4 s 机械/电气端口 RMSE','单事件纯推理 s','终态']
    print('## 绝对角参考下的固定末点\n\n'+table(cols,nominal))
    print('\n## 0.01 历史噪声敏感性\n\n'+table(cols,noisy))
    print('\n## 模型复杂度与功率\n\n'+table(['系统','方法','训练样本数','接受更新','全参数','可训练参数',
                                                 '演化状态','链式功率最大残差','功率最小','功率最大'],physical))


if __name__=='__main__':
    render()
