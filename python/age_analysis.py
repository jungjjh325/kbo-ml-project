# age_analysis.py  (가설 2)
# 연령대 정보를 넣기 전후 비교 + 연령대별 변화 분석.
#  Part 1: 연령대별 기술통계
#  Part 2: 회귀 (나이 없이 / 나이 넣고) 과거 시즌별 롤링 MSE 비교
#  Part 3: 상승/비상승 로지스틱 (IRLS, classify.py 재사용) 나이 넣기 전후 Accuracy
#  Part 4: 회귀 트리(8주차)로 나이 분기점 찾기
# 기준 구현 reference/age_analysis_ref.py 와 같은 결과를 내도록 작성했다.

import os
import numpy as np
import pandas as pd
from classify import irls, sigmoid

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'data')
OUT_DIR = os.path.join(HERE, '..', 'output')

# TODO(확인 필요): 연령대 시작 나이와 폭(3년)은 팀이 정한 값이며 수업자료에 근거가 없다.
BANDS = [(22, 24), (25, 27), (28, 30), (31, 33), (34, 36), (37, 40)]
AGE_MIN, AGE_MAX = 22, 40       # 다음 해 나이 범위
MIN_BAND = 30                   # 연령대 학습 행이 이보다 적으면 전체 모델 사용
MIN_TRAIN = 100                 # 학습 행이 이보다 적으면 그 연도 건너뜀
# TODO(확인 필요): 트리 깊이 2, 잎 최소 40행은 수업자료에 없는 임의값. 바꾸면 깊이 1~3, 잎 20~80 조합으로 결론을 다시 확인한다.
TREE_DEPTH = 2
MIN_LEAF = 40
EVAL_YEARS = range(2015, 2026)  # 직전 시즌 t (정답은 t+1 = 2016~2026)
FIRST_TRAIN = 2010
MODELS_REG = ['직선', '연령대별직선', '직선+나이', '연령대평균변화']
MODELS_CLS = [('모델1', ['v']), ('모델2', ['v', 'ch']), ('모델3_나이', ['v', 'ch', 'age'])]


def band_of(age):
    for i in range(len(BANDS)):
        if BANDS[i][0] <= age <= BANDS[i][1]:
            return i
    return -1


def band_label(i):
    return str(BANDS[i][0]) + '-' + str(BANDS[i][1])


def load_data():
    hit = pd.read_csv(os.path.join(DATA_DIR, 'kbo_hitters_season_1982_2026.csv'), dtype={'playerID': str})
    pit = pd.read_csv(os.path.join(DATA_DIR, 'kbo_pitchers_season_1982_2026.csv'), dtype={'playerID': str})
    people = pd.read_csv(os.path.join(DATA_DIR, 'kbo_people.csv'), dtype={'playerID': str})
    people = people[['playerID', 'birthYear']]
    hit['HRPA'] = hit['HR'] / hit['PA']   # PA >= 100 행만 쓰므로 0 나누기 없음
    pit['IP'] = pit['IPouts'] / 3
    pit['K9'] = pit['SO'] * 9 / pit['IP']  # IP >= 20 행만 쓰므로 0 나누기 없음
    return hit, pit, people


def make_pairs(df, people, vol, limit, metric):
    # 한 행 = 한 선수의 연속한 두 시즌 (season = 직전 시즌 t)
    d = df.merge(people, on='playerID')
    d = d.dropna(subset=['birthYear']).copy()          # 생년 없는 선수 제외
    d['age'] = d['season'] - d['birthYear'].astype(int)

    cur = d[['playerID', 'season', 'age', vol]].copy()
    cur['v'] = d[metric].values

    nxt = cur.rename(columns={'age': 'age_n', vol: 'vol_n', 'v': 'nx'})
    nxt['season'] = nxt['season'] - 1                   # t+1 기록을 t 행에 붙이기 위해

    prev = cur[['playerID', 'season', 'v']].rename(columns={'v': 'vprev'})
    prev['season'] = prev['season'] + 1                 # t-1 기록을 t 행에 붙이기 위해

    m = cur.merge(nxt, on=['playerID', 'season'])
    m = m.merge(prev, on=['playerID', 'season'], how='left')
    m = m.dropna(subset=['v', 'nx'])
    m = m[(m[vol] >= limit) & (m['vol_n'] >= limit)]
    m = m[(m['age_n'] >= AGE_MIN) & (m['age_n'] <= AGE_MAX)].copy()

    m['band'] = m['age_n'].apply(band_of)
    m['y'] = (m['nx'] > m['v']).astype(float)          # 같으면 0
    m['dvol'] = m['vol_n'] - m[vol]
    m['ch'] = m['v'] - m['vprev']
    return m


def fit_line(X, y):
    # 절편 포함 최소제곱 (가우시안 MLE)
    Phi = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(Phi, y, rcond=None)[0]


def predict_line(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


# ---------- Part 4: 회귀 트리 (8주차) ----------
def build_tree(x, y, idx, depth):
    # 반환: ('leaf', 평균, n) 또는 ('split', 분기점, 왼쪽, 오른쪽). 왼쪽 = x <= 분기점
    if depth == 0 or len(idx) < 2 * MIN_LEAF:
        return ('leaf', y[idx].mean(), len(idx))
    best = None
    values = np.unique(x[idx])
    for s in values[:-1]:
        left = idx[x[idx] <= s]
        right = idx[x[idx] > s]
        if len(left) < MIN_LEAF or len(right) < MIN_LEAF:
            continue
        sse = ((y[left] - y[left].mean()) ** 2).sum() + ((y[right] - y[right].mean()) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, s, left, right)
    if best is None:
        return ('leaf', y[idx].mean(), len(idx))
    return ('split', best[1], build_tree(x, y, best[2], depth - 1), build_tree(x, y, best[3], depth - 1))


def tree_leaves(node, lo, hi):
    if node[0] == 'leaf':
        return [(lo, hi, node[1], node[2])]
    cut = int(node[1])
    return tree_leaves(node[2], lo, cut) + tree_leaves(node[3], cut + 1, hi)


def fit_tree(rows, depth):
    x = rows['age_n'].values.astype(float)
    y = (rows['nx'] - rows['v']).values
    return build_tree(x, y, np.arange(len(x)), depth)


def main():
    hit, pit, people = load_data()
    config = [('타자', hit, 'PA', 100, ['AVG', 'SLG', 'HRPA', 'H', 'HR']),
              ('투수', pit, 'IP', 20, ['K9', 'SO', 'BB', 'IP'])]

    desc, reg, cls, leaves, stab = [], [], [], [], []

    for group, df, vol, limit, metrics in config:
        for metric in metrics:
            m = make_pairs(df, people, vol, limit, metric)

            # ---- Part 1: 목표 연도 2011~2026 (t >= 2010) ----
            mm = m[m['season'] >= FIRST_TRAIN]
            for bi in sorted(mm['band'].unique()):
                g = mm[mm['band'] == bi]
                change = g['nx'] - g['v']
                desc.append({'group': group, 'metric': metric, 'band': band_label(bi), 'n': len(g),
                             'mean_change': change.mean(), 'sd_change': change.std(),
                             'up_rate': g['y'].mean(), 'mean_dvol': g['dvol'].mean()})

            # ---- Part 2, 3: 롤링 평가 (학습은 2010 <= t' < t) ----
            for t in EVAL_YEARS:
                tr = m[(m['season'] >= FIRST_TRAIN) & (m['season'] < t)]
                te = m[m['season'] == t]
                if len(te) == 0 or len(tr) < MIN_TRAIN:
                    continue

                y = te['nx'].values
                v = te['v'].values
                sse_base = ((y - v) ** 2).sum()                       # 기준: 직전 시즌 그대로

                b_line = fit_line(tr[['v']].values, tr['nx'].values)
                p_line = predict_line(b_line, te[['v']].values)       # 직선
                b_age = fit_line(tr[['v', 'age_n']].values, tr['nx'].values)
                p_age = predict_line(b_age, te[['v', 'age_n']].values)  # 직선+나이

                all_change = (tr['nx'] - tr['v']).mean()
                p_band = np.zeros(len(te))                            # 연령대별 직선
                p_bmean = np.zeros(len(te))                           # 연령대 평균 변화
                for bi in range(len(BANDS)):
                    ix = (te['band'] == bi).values
                    if not ix.any():
                        continue
                    tb = tr[tr['band'] == bi]
                    if len(tb) >= MIN_BAND:
                        p_band[ix] = predict_line(fit_line(tb[['v']].values, tb['nx'].values), te.loc[ix, ['v']].values)
                        p_bmean[ix] = v[ix] + (tb['nx'] - tb['v']).mean()
                    else:
                        p_band[ix] = p_line[ix]
                        p_bmean[ix] = v[ix] + all_change

                preds = [p_line, p_band, p_age, p_bmean]
                for k in range(4):
                    reg.append({'group': group, 'metric': metric, 'target_year': t + 1, 'model': MODELS_REG[k],
                                'n': len(te), 'sse': ((y - preds[k]) ** 2).sum(), 'sse_base': sse_base})

                # Part 3: vprev 가 있는 행만
                trc = tr.dropna(subset=['vprev'])
                tec = te.dropna(subset=['vprev'])
                if len(tec) == 0 or len(trc) < MIN_TRAIN:
                    continue
                yt = tec['y'].values
                if trc['y'].mean() > 0.5:     # 다수 클래스 (동률이면 0)
                    majority = 1.0
                else:
                    majority = 0.0
                rec = {'group': group, 'metric': metric, 'target_year': t + 1, 'n': len(tec),
                       'correct_base': float((yt == majority).sum())}
                for name, cols in MODELS_CLS:
                    beta, conv, it = irls(trc[cols].values, trc['y'].values)
                    p = sigmoid(np.column_stack([np.ones(len(tec)), tec[cols].values]) @ beta)
                    rec['correct_' + name] = float(((p >= 0.5) == yt).sum())
                cls.append(rec)

            # ---- Part 4: 회귀 트리, 2010 <= t <= 2024 학습 ----
            tr = m[(m['season'] >= FIRST_TRAIN) & (m['season'] <= 2024)]
            for lo, hi, mu, n in tree_leaves(fit_tree(tr, TREE_DEPTH), AGE_MIN, AGE_MAX):
                leaves.append({'group': group, 'metric': metric, 'age_from': lo, 'age_to': hi, 'mean_change': mu, 'n': n})

            # 안정성: 학습 종료 연도 2014~2024, 깊이 1 분기점 (분기점 이하 / 초과)
            cuts = []
            for end in range(2014, 2025):
                tr = m[(m['season'] >= FIRST_TRAIN) & (m['season'] <= end)]
                node = fit_tree(tr, 1)
                if node[0] == 'split':
                    cuts.append(int(node[1]))
                else:
                    cuts.append(None)
            stab.append({'group': group, 'metric': metric, 'cuts_2014_2024': str(cuts)})

    # ---- 저장 ----
    pd.DataFrame(desc).to_csv(os.path.join(OUT_DIR, 'age_band_stats.csv'), index=False, encoding='utf-8-sig')

    R = pd.DataFrame(reg)
    R.to_csv(os.path.join(OUT_DIR, 'age_regression_by_year.csv'), index=False, encoding='utf-8-sig')
    S = R.groupby(['group', 'metric', 'model']).agg(n=('n', 'sum'), sse=('sse', 'sum'), sse_base=('sse_base', 'sum')).reset_index()
    S['mse_ratio_vs_last'] = S['sse'] / S['sse_base']
    line_sse = S[S['model'] == '직선'].set_index(['group', 'metric'])['sse']
    S['mse_ratio_vs_line'] = S['sse'] / line_sse.loc[list(zip(S['group'], S['metric']))].values   # 모델 2(직선) 대비
    S.to_csv(os.path.join(OUT_DIR, 'age_regression_summary.csv'), index=False, encoding='utf-8-sig')

    C = pd.DataFrame(cls)
    C.to_csv(os.path.join(OUT_DIR, 'age_logistic_by_year.csv'), index=False, encoding='utf-8-sig')
    CS = C.groupby(['group', 'metric']).sum(numeric_only=True).reset_index()
    acc_cols = []
    for k in ['base', '모델1', '모델2', '모델3_나이']:
        CS['acc_' + k] = CS['correct_' + k] / CS['n']
        acc_cols.append('acc_' + k)
    CS[['group', 'metric', 'n'] + acc_cols].to_csv(os.path.join(OUT_DIR, 'age_logistic_summary.csv'), index=False, encoding='utf-8-sig')

    pd.DataFrame(leaves).to_csv(os.path.join(OUT_DIR, 'age_tree_leaves.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(stab).to_csv(os.path.join(OUT_DIR, 'age_tree_stability.csv'), index=False, encoding='utf-8-sig')

    print(S.pivot(index=['group', 'metric'], columns='model', values='mse_ratio_vs_last').round(3).to_string())
    print(CS[['group', 'metric', 'n'] + acc_cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
