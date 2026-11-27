"""Render the fixed Sprint 2A symmetric comparison from saved results only."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.osgkd_v2_refits.table3 import table
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io


RUN = 'osgkd_v2_wp3_2_flat_seeds_20261128'
SEEDS = (24111, 24121, 24131)
SCOPES = tuple(f'stable_{h}s' for h in (1, 2, 5, 8)) + ('unstable_actual',)


def fmt(value, digits=4):
    return f'{value:.{digits}f}' if value is not None else '—'


def aggregate(item, digits=4):
    if item['mean'] is None:
        return f"— (n={item['n']})"
    return fmt(item['mean'], digits) + ' ± ' + fmt(item['sample_sd'], digits)


def render():
    score = io.read(ROOT / 'results' / RUN / 'evaluate01' / 'summary.json')
    io.require(score['run_id'] == RUN and score['original_val_passes'] == 1, 'once-scored val receipt required')
    t = score['comparison']
    risk, physical, spectral, pilot, paired = [], [], [], [], []
    for variant, label in (('graph', '图'), ('flat', '扁平')):
        for seed in SEEDS:
            name = f'{variant}_{seed}'
            row, time = t['per_endpoint'][name], score['timings'][name]
            risk.append([name, f"{row['qualified']}/107", *[fmt(row[s + '_risk']) for s in SCOPES[:4]],
                         f"{row['stable_false_alarms']}/95",
                         f"{row['unstable_autonomous_hits']}/12 · {row['unstable_autonomous_hits']}/22",
                         f"{fmt(row['unstable_actual_risk'])} ({row['unstable_actual_covered']}/12)",
                         f"{row['unstable_actual_hits']}/12 · {row['unstable_actual_hits']}/22"])
            physical.append([name, *[' / '.join(fmt(row[s + '_' + q]) for q in ('speed', 'tau_m', 'tau_e')) for s in SCOPES]])
            spectral.append([name, row['lag_count'], fmt(row['lag_median_s'], 3), fmt(row['lag_iqr_s'], 3),
                             f"{fmt(row['lag_q1_s'],3)} / {fmt(row['lag_q3_s'],3)}", fmt(row['alpha'], 6),
                             f"{row['positive_modes']}/49",
                             ' / '.join(fmt(time[k], 3) for k in ('initialize_s','rk4_0_005_s','rk4_0_0025_s')),
                             fmt(row['inference_s'], 3)])
            p = t['pilot_0232'][name]
            categories = {'autonomous_trigger': '自主失步', 'history_trigger': '历史内触发',
                          'no_trigger': '无触发', 'numerical_unresolved': '数值未决',
                          'input_uncovered': '未覆盖', 'latency_uncovered': '时延未覆盖'}
            pilot.append([name, categories.get(p['outcome'], p['outcome']), fmt(p['predicted_record_s'], 3),
                          *[fmt(p['event_mean_motion_error'][str(h)]) for h in (1, 2, 5, 8)],
                          fmt(p['sensitivity_stable_8s_risk'])])
        st = t['group_statistics'][variant]
        title = f'{label}均值 ± 样本 SD'
        risk.append([title, aggregate(st['qualified'], 2), *[aggregate(st[s + '_risk']) for s in SCOPES[:4]],
                     aggregate(st['stable_false_alarms'], 2), aggregate(st['unstable_autonomous_hits'], 2),
                     aggregate(st['unstable_actual_risk']), aggregate(st['unstable_actual_hits'], 2)])
        physical.append([title, *[' / '.join(aggregate(st[s + '_' + q]) for q in ('speed', 'tau_m', 'tau_e')) for s in SCOPES]])
        spectral.append([title, aggregate(st['lag_count'], 2), aggregate(st['lag_median_s'], 3),
                         aggregate(st['lag_iqr_s'], 3),
                         ' / '.join(aggregate(st[k], 3) for k in ('lag_q1_s','lag_q3_s')),
                         aggregate(st['alpha'], 6), aggregate(st['positive_modes'], 2), '—', aggregate(st['inference_s'], 3)])
    for seed in SEEDS:
        by = t['graph_minus_flat'][str(seed)]
        lag = t['common_trigger_graph_minus_flat_s'][str(seed)]
        within = lag['actual_comparable']
        paired.append([seed, fmt(by['stable_8s_risk']), fmt(by['stable_false_alarms'], 0),
                       fmt(by['unstable_autonomous_hits'], 0), fmt(by['unstable_actual_risk']),
                       f"{lag['common']} ({lag['count']})", f"{lag['graph_only']} / {lag['flat_only']}",
                       fmt(lag['median'], 3), f"{fmt(lag['q1'],3)} / {fmt(lag['q3'],3)}", fmt(lag['iqr'],3),
                       within['count'], fmt(within['median'], 3)])
    st = t['graph_minus_flat_statistics']
    print('## Table 1A · 轨迹风险与失稳检出\n\n' + table(
        ['种子方案','资格','稳 1 s','稳 2 s','稳 5 s','稳 8 s','误报 /95','8 s 自主召回 /12 与 /22',
         '真实区间风险（覆盖）','真实区间检出 /12 与 /22'], risk))
    print('\n## Table 1B · 共同转速和端口\n\n' + table(
        ['种子方案','稳定 1 s','稳定 2 s','稳定 5 s','稳定 8 s','失稳真实区间'], physical))
    print('\n每格顺序为共同转速 RMSE（rad/s）、归一化机械和归一化电气端口 RMSE。')
    print('\n## Table 1C · 滞后、固定参考谱和纯推理\n\n' + table(
        ['种子方案','有效滞后数','滞后中位 s','IQR s','Q1 / Q3 s','α s⁻¹','正模态',
         '初始化 / 主步长 / 细步长 s','纯推理合计 s'], spectral))
    print('\n## Table 1D · 同种子图减扁平及共同触发时差\n\n' + table(
        ['种子','稳 8 s 风险差','误报数差','自主召回数差','真实风险差','共同触发（可比）',
         '图独有 / 扁平独有','共同滞后差中位 s','Q1 / Q3 s','IQR s','双方真实区间可比数',
         '真实区间共同滞后差中位 s'], paired))
    print('\n8 s 共同滞后要求双方都自主触发且同一真实触发时刻已知。双方真实区间可比列再要求两个预测触发都在真实记录内。')
    names = {'stable_8s_risk':'稳定 8 s 风险', 'stable_false_alarms':'稳定误报数',
             'unstable_autonomous_hits':'自主召回数', 'unstable_actual_risk':'真实区间风险'}
    print('\n' + table(['图减扁平','配对差均值 ± 样本 SD','最小 / 最大','图低 / 相同 / 图高'], [
        [title, aggregate(st[k]), fmt(st[k]['minimum'])+' / '+fmt(st[k]['maximum']),
         f"{st[k]['lower']} / {st[k]['equal']} / {st[k]['higher']}"] for k,title in names.items()
    ]))
    print('\n## pilot_0232 · 固定边界样本诊断\n\n' + table(
        ['种子方案','8 s 分类','触发时刻 s（记录）','1 s','2 s','5 s','8 s','省去该样本后的稳定 8 s 风险 /94'], pilot))
    print('\n1/2/5/8 s 列为该事件的平均归一化运动平方误差。/94 风险是敏感性诊断，Table 1A 的正式稳定分母为 95。')


if __name__ == '__main__':
    render()
