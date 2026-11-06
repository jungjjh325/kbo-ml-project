# make_web_data.py
# output/ 의 CSV를 읽어 화면용 web/data.js 를 만든다.
# 화면(index.html)은 계산을 하지 않으므로, 필요한 값은 모두 여기서 넣어 둔다.

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'output')
WEB_DIR = os.path.join(HERE, '..', 'web')

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def to_num(v):
    # NaN 은 JSON 에서 null 로 쓰기 위해 None 으로 바꾼다
    if pd.isna(v):
        return None
    return float(v)


def sort_key(p):
    return (p['kind'], p['name'])


def logistic_data():
    # 가설 3 (classify.py 결과). results: logistic_results.csv 그대로
    # p: group -> playerID -> metric -> {p1, p2, pred2, label}  (타자는 hit50/hit100에서 확률이 다름)
    res = pd.read_csv(os.path.join(OUT_DIR, 'logistic_results.csv'))
    lp = pd.read_csv(os.path.join(OUT_DIR, 'logistic_predictions.csv'), dtype={'playerID': str})

    results = []
    for _, r in res.iterrows():
        row = {}
        for c in res.columns:
            v = r[c]
            if c in ('group', 'metric'):
                row[c] = v
            elif c.startswith('converged'):
                row[c] = bool(v)
            else:
                row[c] = float(v)
        results.append(row)

    p = {}
    for _, r in lp.iterrows():
        g = p.setdefault(r['group'], {})
        one = g.setdefault(r['playerID'], {})
        label = None
        if pd.notna(r['label2026']):
            label = int(r['label2026'])
        one[r['metric']] = {'p1': float(r['p_up_model1']), 'p2': float(r['p_up_model2']),
                            'pred2': int(r['pred_model2']), 'label': label}
    return {'results': results, 'p': p}


def csv_records(name):
    # CSV 를 행 목록으로 (NaN -> null)
    df = pd.read_csv(os.path.join(OUT_DIR, name))
    return df.astype(object).where(df.notna(), None).to_dict('records')


def age_data():
    # 가설 2 (age_analysis.py 결과). CSV 값 그대로 넣는다.
    return {'bands': csv_records('age_band_stats.csv'),
            'reg': csv_records('age_regression_summary.csv'),
            'cls': csv_records('age_logistic_summary.csv'),
            'leaves': csv_records('age_tree_leaves.csv'),
            'stab': csv_records('age_tree_stability.csv')}


def main():
    pred = pd.read_csv(os.path.join(OUT_DIR, 'player_predictions.csv'), dtype={'playerID': str})
    summary = pd.read_csv(os.path.join(OUT_DIR, 'results_summary_check.csv'))
    sizes = pd.read_csv(os.path.join(OUT_DIR, 'group_sizes.csv'))

    hit = pd.read_csv(os.path.join(DATA_DIR, 'kbo_hitters_season_1982_2026.csv'), dtype={'playerID': str})
    pit = pd.read_csv(os.path.join(DATA_DIR, 'kbo_pitchers_season_1982_2026.csv'), dtype={'playerID': str})
    pit['IP'] = pit['IPouts'] / 3

    # 2025 소속 팀, 2026 출전량(타자 PA, 투수 IP)
    team = {}
    for _, r in hit[hit['season'] == 2025].iterrows():
        team[r['playerID']] = r['team']
    for _, r in pit[pit['season'] == 2025].iterrows():
        team[r['playerID']] = r['team']

    vol26 = {'hit': {}, 'pit': {}}
    for _, r in hit[hit['season'] == 2026].iterrows():
        vol26['hit'][r['playerID']] = r['PA']
    for _, r in pit[pit['season'] == 2026].iterrows():
        vol26['pit'][r['playerID']] = r['IP']

    # groups
    groups = {}
    for _, r in sizes.iterrows():
        groups[r['group']] = {'n6': int(r['six_season_and_threshold']),
                              'n26': int(r['with_2026_record']), 'league': {}}
    for _, r in summary.iterrows():
        groups[r['group']]['league'][r['metric']] = float(r['league_mean'])
    # 평가 제외 사유별 인원: 2026 행 없음 / 2026 행은 있으나 출전 0
    for g in groups:
        v = vol26['hit' if g.startswith('hit') else 'pit']
        ids = set(pred.loc[pred['group'] == g, 'playerID'])
        groups[g]['no26'] = sum(1 for pid in ids if pid not in v)
        groups[g]['zero26'] = sum(1 for pid in ids if pid in v and v[pid] == 0)

    # players
    players = {}
    for _, r in pred.iterrows():
        pid = r['playerID']
        if r['group'].startswith('hit'):
            kind = 'hit'
        else:
            kind = 'pit'

        if pid not in players:
            players[pid] = {'id': pid, 'name': r['name'], 'kind': kind, 'team': team.get(pid),
                            'groups': [], 'vol26': to_num(vol26[kind].get(pid, np.nan)), 'm': {}}
        p = players[pid]
        if r['group'] not in p['groups']:
            p['groups'].append(r['group'])

        y = []
        for year in YEARS:
            y.append(to_num(r['y' + str(year)]))
        pred_v = float(r['pred_line'])
        act = to_num(r['actual2026'])
        if act is None:
            actdiff = None
        else:
            actdiff = act - pred_v

        p['m'][r['metric']] = {
            'y': y,
            'pred': pred_v,
            'own': float(r['pred_own_mean']),
            'act': act,
            'line0': float(r['line_2020']),
            'diff': pred_v - y[5],      # 예측 - 2025
            'neg': bool(pred_v < 0),    # 0 미만 표시용
            'actdiff': actdiff,         # 실제 - 예측
        }

    # ratios: CSV 값 그대로
    ratios = []
    for _, r in summary.iterrows():
        ratios.append({'g': r['group'], 'm': r['metric'], 'n': int(r['n']),
                       'last': float(r['ratio_last']), 'own': float(r['ratio_own']),
                       'lg': float(r['ratio_league'])})

    # 타자 먼저, 그다음 이름 순
    player_list = sorted(players.values(), key=sort_key)
    data = {'groups': groups, 'players': player_list, 'ratios': ratios, 'logistic': logistic_data(), 'age': age_data()}

    with open(os.path.join(WEB_DIR, 'data.js'), 'w', encoding='utf-8') as f:
        f.write('window.KBO_DATA = ' + json.dumps(data, ensure_ascii=False) + ';\n')

    print('data.js 저장: 선수', len(player_list), '명, 비교', len(ratios), '행')


if __name__ == '__main__':
    main()
