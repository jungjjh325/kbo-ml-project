# predict.py
# 선수마다 2020~2025 기록에 직선(1차 선형회귀)을 맞춰 2026을 예측하고,
# 기준모델 3개(직전 시즌, 본인 평균, 전체 평균)와 MSE로 비교한다.

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'output')

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
X = np.array([0, 1, 2, 3, 4, 5])  # 2020 -> 0, ..., 2025 -> 5

# (group 이름, 타자/투수, 출전량 컬럼, 기준값, 지표 목록)
GROUPS = [
    ('hit50', 'hit', 'PA', 50, ['AVG', 'SLG', 'HRPA', 'H', 'HR']),
    ('hit100', 'hit', 'PA', 100, ['AVG', 'SLG', 'HRPA', 'H', 'HR']),
    ('pit20', 'pit', 'IP', 20, ['K9', 'SO', 'BB', 'IP']),
]


def load_data():
    hit = pd.read_csv(os.path.join(DATA_DIR, 'kbo_hitters_season_1982_2026.csv'), dtype={'playerID': str})
    pit = pd.read_csv(os.path.join(DATA_DIR, 'kbo_pitchers_season_1982_2026.csv'), dtype={'playerID': str})

    # 파생 지표: 분모가 0이면 NaN
    hit['HRPA'] = np.nan
    ok = hit['PA'] > 0
    hit.loc[ok, 'HRPA'] = hit.loc[ok, 'HR'] / hit.loc[ok, 'PA']

    pit['IP'] = pit['IPouts'] / 3
    pit['K9'] = np.nan
    ok = pit['IP'] > 0
    pit.loc[ok, 'K9'] = pit.loc[ok, 'SO'] * 9 / pit.loc[ok, 'IP']
    return hit, pit


def mse(pred, actual):
    return ((pred - actual) ** 2).mean()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hit, pit = load_data()

    rows = []   # player_predictions.csv
    sizes = []  # group_sizes.csv

    for group, kind, vol, limit, metrics in GROUPS:
        if kind == 'hit':
            df = hit
        else:
            df = pit

        # 표(행=선수, 열=시즌)로 바꾼다
        recent = df[(df['season'] >= 2020) & (df['season'] <= 2026)]
        vol_table = recent.pivot(index='playerID', columns='season', values=vol)

        # 대상: 2020~2025 매년 기록이 있고 출전량 >= 기준
        targets = []
        for pid in vol_table.index:
            v = vol_table.loc[pid, YEARS]
            if v.notna().all() and (v >= limit).all():
                targets.append(pid)

        # 2026 기록 있음 = 2026 행이 있고 출전량 > 0 (출전 0인 행은 기록 없음으로 본다)
        played_2026 = set()
        if 2026 in vol_table.columns:
            for pid in targets:
                if vol_table.loc[pid, 2026] > 0:
                    played_2026.add(pid)
        with_2026 = len(played_2026)
        sizes.append({'group': group, 'six_season_and_threshold': len(targets), 'with_2026_record': with_2026})

        # 선수 이름: 가장 최근 시즌 이름
        names = df.sort_values('season').groupby('playerID')['name'].last()

        # 전체 평균: 기준을 넘긴 2020~2025 모든 선수-시즌 행의 평균
        pool = df[(df['season'] >= 2020) & (df['season'] <= 2025) & (df[vol] >= limit)]

        for metric in metrics:
            table = recent.pivot(index='playerID', columns='season', values=metric)
            league_mean = pool[metric].mean()

            for pid in targets:
                y = table.loc[pid, YEARS].values.astype(float)
                if np.isnan(y).any():
                    continue

                a, b = np.polyfit(X, y, 1)  # y = a*x + b
                actual = np.nan
                if pid in played_2026:
                    actual = table.loc[pid, 2026]

                row = {'group': group, 'metric': metric, 'playerID': pid, 'name': names[pid]}
                for i in range(6):
                    row['y' + str(YEARS[i])] = y[i]
                row['line_2020'] = b              # x = 0
                row['pred_line'] = a * 6 + b      # x = 6 (2026), 0으로 자르지 않음
                row['pred_last'] = y[5]
                row['pred_own_mean'] = y.mean()
                row['pred_league_mean'] = league_mean
                row['actual2026'] = actual
                if pd.notna(actual):
                    row['in_eval'] = 1
                else:
                    row['in_eval'] = 0
                rows.append(row)

    pred = pd.DataFrame(rows)

    # 평가: 2026에 출전했고 해당 지표 값이 있는 선수만
    summary = []
    ev = pred[pred['in_eval'] == 1]
    for group, kind, vol, limit, metrics in GROUPS:
        for metric in metrics:
            s = ev[(ev['group'] == group) & (ev['metric'] == metric)]
            m_line = mse(s['pred_line'], s['actual2026'])
            m_last = mse(s['pred_last'], s['actual2026'])
            m_own = mse(s['pred_own_mean'], s['actual2026'])
            m_lg = mse(s['pred_league_mean'], s['actual2026'])
            summary.append({'group': group, 'metric': metric, 'n': len(s),
                            'ratio_last': m_line / m_last, 'ratio_own': m_line / m_own,
                            'ratio_league': m_line / m_lg, 'league_mean': s['pred_league_mean'].iloc[0]})
    summary = pd.DataFrame(summary)
    sizes = pd.DataFrame(sizes)

    pred.to_csv(os.path.join(OUT_DIR, 'player_predictions.csv'), index=False, encoding='utf-8-sig')
    summary.to_csv(os.path.join(OUT_DIR, 'results_summary_check.csv'), index=False, encoding='utf-8-sig')
    sizes.to_csv(os.path.join(OUT_DIR, 'group_sizes.csv'), index=False, encoding='utf-8-sig')

    print(sizes.to_string(index=False))
    print(summary.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
