"""Render recovery and transfer tables from the frozen Layer 1 score receipts."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.osgkd_v2_refits.table3 import f, table
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io


RUN = 'osgkd_v2_wp2_layer1_20261127'
PUBLIC = ROOT/'results'/RUN
LABELS = {'osgkd':'Open OSGKD','static_c0':'静态 C0','neural_ode':'图 Neural ODE',
          'closed_passive':'封闭被动 Kuramoto','weak_static':'弱式稀疏静态',
          'same_template':'同模板从头拟合','ode_adapt':'冻结编码器、适配 ODE','frozen_lift':'完全冻结映射'}


def forecast_columns(endpoint, raw):
    if not endpoint.get('evaluated'):
        return ['—']*8 + [endpoint['fit']['status']]
    forecast = endpoint['forecast']
    coverage = f"{forecast['coverage']}/{raw}"
    metrics = [f((forecast['windows'][str(h)] or {}).get('risk_mean'),5) for h in (100,200,400)]
    last = forecast['windows']['400']
    vector = last['rmse'] if last else {}
    omega = ' / '.join(f(vector.get(k)) for k in ('theta','omega'))
    ports = ' / '.join(f(vector.get(k)) for k in ('tau_m','tau_e'))
    return [coverage,' / '.join(metrics),f((last or {}).get('risk_mean_tail'),5),omega,ports,f((last or {}).get('common_speed_rmse')),
            f(forecast['inference_per_event_s'],5),endpoint['fit']['accepted'],endpoint['fit']['status']]


def recovery_columns(endpoint):
    data = endpoint.get('structure')
    if data is None:
        return ['—']*6
    precision = [' / '.join(f(data[group][stat],3) for stat in ('precision','recall','f1'))
                 for group in ('edges','beta')]
    excitation = data['resolved_by_excitation']
    strength = f(excitation['c_relative_l2'],3)+' / '+f(excitation['beta_relative_l2_in_true_a_coordinate'],3)
    active = endpoint['nonzero_ode_edge_counts']
    counts = f"{data['edges']['predicted_count']} / {active.get('c',0)}"
    return [*precision,f(data['modulation_accuracy'],3),strength,f(excitation['same_state_normalized_node_force_rmse'],5),counts]


def render():
    released = io.read(PUBLIC/'release.json')
    scored = io.read(PUBLIC/'evaluate01/summary.json')
    fit = io.read(PUBLIC/'fit_summary.json')
    case_rows, recovery_rows, transfer_rows, transfer_structure, optimization = [], [], [], [], []
    for job in released['jobs']:
        n, c, noise, seed = (job[k] for k in ('n','condition','noise','seed'))
        key = f'n{n}_{c}_noise{noise}_s{seed}'
        cell = io.read(io.verified(scored['cells'][key]))
        io.require(cell['job']==job and cell['val_count']==24,'fixed cell dimensions differ')
        prefix = [c,noise,seed] if n==8 else [n,seed]
        for method in job['methods']:
            r = cell['methods'][method]
            prediction = case_rows if n==8 else transfer_rows
            structure = recovery_rows if n==8 else transfer_structure
            if 'dense' in r:
                d = r['dense']
                optimization.append([key,method,'密训',d['accepted'],d.get('clipped','—'),
                                     f(d.get('initial_fixed_batch_loss'),5),f(d.get('final_fixed_batch_loss'),5),
                                     f(d.get('first32_horizon_mean_loss'),5),f(d.get('last32_horizon_mean_loss'),5)])
            if not r['variants']:
                prediction.append([*prefix,LABELS[method],*forecast_columns({'fit':r['dense']},24)])
            for variant,v in r['variants'].items():
                label = LABELS[method]+(' · 剪枝' if variant=='sparse' else '')
                prediction.append([*prefix,label,*forecast_columns(v,24)])
                structure.append([*prefix,label,*recovery_columns(v)])
                d = v['fit']
                optimization.append([key,method,variant,d['accepted'],d.get('clipped','—'),
                                     f(d.get('initial_fixed_batch_loss'),5),f(d.get('final_fixed_batch_loss'),5),
                                     f(d.get('first32_horizon_mean_loss'),5),f(d.get('last32_horizon_mean_loss'),5)])
    forecast = ['方法','覆盖','1/2/4 s 事件均值风险','4 s 尾权风险','4 s 角度/转速 RMSE','4 s 机械/电气 RMSE',
                '4 s 共同转速 RMSE','单事件纯推理 s','该阶段接受更新','终态']
    support = ['方法','c 精确率/召回/F1','β 精确率/召回/F1','调制准确率','受激励 c/β 强度相对误差',
               '同状态节点力 RMSE','c 估计支撑/实际非零边']
    print('## 八节点同信息预测\n\n'+table(['工况','历史噪声','种子',*forecast],case_rows))
    print('\n## 八节点真值结构恢复\n\n'+table(['工况','历史噪声','种子',*support],recovery_rows))
    print('\n## 有节点角色映射的跨规模预测\n\n'+table(['目标节点数','源种子',*forecast],transfer_rows))
    print('\n## 跨规模真值结构恢复\n\n'+table(['目标节点数','源种子',*support],transfer_structure))
    print('\n## 固定训练诊断\n\n'+table(['格','方法','阶段','接受','裁剪','固定批初始损失',
                                               '固定批末点损失','首 32 批均值','末 32 批均值'],optimization))
    print(f"\n拟合进程墙钟 {fit['wall_s']/3600:.2f} h，CPU {fit['cpu_s']/3600:.2f} h。")


if __name__=='__main__':
    render()
