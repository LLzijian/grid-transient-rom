"""Render Table 3 from frozen WP1A receipts without opening a dataset."""

from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io


RUN = 'osgkd_v2_wp1a_refits_20261127'
PUBLIC = ROOT/'results'/RUN
SPANS = ('stable_1s','stable_2s','stable_5s','stable_8s','unstable_actual')


def table(headers, rows):
    all_rows = [headers, ['---']*len(headers), *rows]
    return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in all_rows)


def f(number, digits=4):
    return '—' if number is None else f'{number:.{digits}f}'


def score_columns(score):
    if 'scopes' not in score:
        return ['—']*13
    risk = [f(score['scopes'][name]['risk']) for name in SPANS]
    auto = score['autonomous']
    fp = auto['stable_false_alarm_8s']['autonomous_trigger']
    rec = auto['unstable_recall_8s']['autonomous_trigger']
    lag, spec = auto['trigger_lag_s'], score['spectrum']
    timing = score['inference']['inference_wall_s']
    return [f"{score['qualified_events']}/107",*risk,f'{fp}/95',f'{rec}/12 · {rec}/22',
            f"{auto['unstable_actual_interval_hits']}/12",f"{f(lag['median'],3)} / {f(lag['iqr'],3)}",
            f(spec.get('max_real'),5),spec.get('positive_modes','—'),f(timing,3)]


def measurement_columns(score):
    if 'scopes' not in score:
        return ['—']*3
    measures = ('common_speed_rmse_rad_s','normalized_tau_m_rmse','normalized_tau_e_rmse')
    return [' / '.join(f(score['scopes'][span][metric]) for span in SPANS) for metric in measures]


def fit_row(data):
    if 'attempts_ref' not in data:
        return [data['attempts'],data['accepted'],data['rejected'],'—','—','—','—','—',data['stop_reason'] or '—']
    attempts = io.read(io.verified(data['attempts_ref']))
    d = io.read(io.verified(data['diagnostics_ref']))
    losses = [row['loss'] for row in attempts if row.get('loss') is not None]
    start = next((record['loss'] for record in d if record['attempt']==0),None)
    end = next((record['loss'] for record in d if record['attempt']==128),None)
    norms = [row['update_norm'] for row in attempts if row.get('update_norm') is not None]
    gh = next((row for row in attempts[::-1] if row.get('gradient_norm') is not None),{})
    return [data['attempts'],data['accepted'],data['rejected'],sum(row.get('clipped') is True for row in attempts),
            f'{f(start)} → {f(end)}',f'{f(mean(losses[:16]) if losses else None)} → {f(mean(losses[-16:]) if losses else None)}',
            f(median(norms) if norms else None,5),f(gh.get('gradient_norm'),3),data['stop_reason'] or '—']


def render():
    fit = io.read(PUBLIC/'fit_summary.json')
    score = io.read(PUBLIC/'evaluate01/summary.json')
    jobs = io.read(io.verified(fit['release_ref']))['jobs']
    io.require(set(fit['jobs'])==set(score['val'])=={j['id'] for j in jobs},'WP1A terminal population differs')
    eval_rows, l1_rows, train_rows, structure_rows = [], [], [], []
    for sid in sorted(score['parents']):
        name = f'parent_{sid}'
        source = score['parents'][sid]
        eval_rows.append([name,*score_columns(source)])
        l1_rows.append([name,*measurement_columns(source)])
        for job in (job for job in jobs if str(job['seed'])==sid):
            jid = job['id']
            source = score['val'][jid]
            receipt = io.read(io.verified(fit['jobs'][jid]['summary_ref']))
            io.require(receipt['job']==job,'terminal job provenance differs')
            eval_rows.append([jid,*score_columns(source)])
            l1_rows.append([jid,*measurement_columns(source)])
            train_rows.append([jid,*fit_row(receipt['result'])])
            if 'complexity' in source:
                c = source['complexity']
                structure_rows.append([jid,c['parameters'],c['embedded_state_dimension'],c['evolved_state_dimension'],
                                       c['constant_mechanical_coordinates'],c['exchange_edges'],c['modulated_edges'],
                                       c['active_edge_paths'],c['active_module_groups']])
    eval_header = ['种子方案','数值资格','稳 1 s','稳 2 s','稳 5 s','稳 8 s','失稳真实区间风险','误报',
                   '自主召回','真实区间检出','滞后中位/IQR s','α s⁻¹','正模态','双步长纯推理 s']
    print('## Table 3 · 固定末点及两套参照\n\n'+table(eval_header,eval_rows))
    print('\n## 同口径端口与共同转速\n\n各行按稳定 1/2/5/8 s 及失稳真实区间排列。\n\n'+table(
          ['种子方案','共同转速 RMSE（rad/s）','归一化机械 RMSE','归一化电气 RMSE'],l1_rows))
    print('\n## 同预算适配健康度\n\n'+table(['组','尝试','接受','拒绝','裁剪','固定首批损失 0→128',
          '前末 16 批平均损失','投影后更新范数中位','末次有限梯度范数','停止原因'],train_rows))
    print('\n## ODE 复杂度\n\n'+table(['组','有效参数','嵌入维','演化维','常量机械坐标','互易边','调制边',
                                            '有效边通道','有效模块'],structure_rows))
    overlap = []
    for variant, pairs in sorted(score['support_overlap'].items()):
        for pair,v in sorted(pairs.items()):
            overlap.append([variant,pair,v.get('first_edges',0),v.get('second_edges',0),v['intersection'],
                            f(v['jaccard']),f(v['f1']),f(v['expected_jaccard']),f(v['expected_f1'])])
    print('\n## 跨种子基准刚度阳性边支撑\n\n'+table(['组','种子对','边数 A','边数 B','交集','Jaccard','F1',
                                                       '随机 Jaccard 期望','随机 F1 期望'],overlap))


if __name__ == '__main__':
    render()
