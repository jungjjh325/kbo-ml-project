# classify.py  (가설 3)
# "다음 시즌 지표가 직전 시즌보다 상승(1)하는가 / 아닌가(0)" 를 지표마다 따로 분류한다.
# 모델: 로지스틱 회귀, 학습: IRLS(뉴턴-랩슨, 7주차 식), numpy로 직접 구현.
# 입력: output/player_predictions.csv (가설 1 결과). 2026 값은 학습에 쓰지 않는다.

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, '..', 'output')

YCOLS = ['y2020', 'y2021', 'y2022', 'y2023', 'y2024', 'y2025']
MAX_ITER = 25
TOL = 1e-8


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def irls(X, t):
    # X: 입력 (n x d), t: 라벨 0/1. 반환: beta, 수렴 여부, 반복 횟수
    Phi = np.column_stack([np.ones(len(X)), X])   # 절편 열 추가
    beta = np.zeros(Phi.shape[1])
    for it in range(1, MAX_ITER + 1):
        p = sigmoid(Phi @ beta)
        r = p * (1 - p)
        r[r < 1e-10] = 1e-10                      # R 대각 원소 바닥
        R = np.diag(r)
        z = Phi @ beta + (t - p) / r              # z = Phi beta + R^-1 (t - p)
        try:
            beta_new = np.linalg.inv(Phi.T @ R @ Phi) @ Phi.T @ R @ z
        except np.linalg.LinAlgError:             # 특이행렬
            return beta, False, it
        if np.abs(beta_new - beta).max() < TOL:
            return beta_new, True, it
        beta = beta_new
    return beta, False, MAX_ITER


def predict_prob(beta, X):
    Phi = np.column_stack([np.ones(len(X)), X])
    return sigmoid(Phi @ beta)


def main():
    pred = pd.read_csv(os.path.join(OUT_DIR, 'player_predictions.csv'), dtype={'playerID': str})

    results = []
    rows = []
    coefs = []

    # group, metric 순서는 가설 1 CSV에 나온 순서 그대로
    pairs = pred[['group', 'metric']].drop_duplicates().values
    for group, metric in pairs:
        s = pred[(pred['group'] == group) & (pred['metric'] == metric)]
        Y = s[YCOLS].values.astype(float)   # (선수 x 6)

        # 학습 행: 직전 시즌 i = 1..4 (2021~2024), 목표 i+1 (2022~2025)
        level, change, label = [], [], []
        for i in range(1, 5):
            level.extend(Y[:, i])
            change.extend(Y[:, i] - Y[:, i - 1])
            label.extend((Y[:, i + 1] > Y[:, i]).astype(int))   # 같으면 0
        level = np.array(level)
        change = np.array(change)
        t_train = np.array(label)

        X1_train = level.reshape(-1, 1)
        X2_train = np.column_stack([level, change])
        b1, conv1, it1 = irls(X1_train, t_train)
        b2, conv2, it2 = irls(X2_train, t_train)

        # 다수 클래스 기준 (학습 라벨, 동률이면 0)
        if t_train.mean() > 0.5:
            majority = 1
        else:
            majority = 0

        # 2026 예측: 모든 선수에 대해 계산 (직전 시즌 2025)
        lv = Y[:, 5]
        ch = Y[:, 5] - Y[:, 4]
        p1 = predict_prob(b1, lv.reshape(-1, 1))
        p2 = predict_prob(b2, np.column_stack([lv, ch]))
        pred1 = (p1 >= 0.5).astype(int)
        pred2 = (p2 >= 0.5).astype(int)

        in_eval = s['in_eval'].values == 1
        actual = s['actual2026'].values
        label26 = np.full(len(s), np.nan)
        label26[in_eval] = (actual[in_eval] > lv[in_eval]).astype(int)

        t_test = label26[in_eval]
        results.append({
            'group': group, 'metric': metric,
            'n_train': len(t_train), 'n_test': int(in_eval.sum()),
            'up_train': t_train.mean(), 'up_test': t_test.mean(),
            'majority_class': majority,
            'acc_baseline': (t_test == majority).mean(),
            'acc_model1': (pred1[in_eval] == t_test).mean(),
            'acc_model2': (pred2[in_eval] == t_test).mean(),
            'converged_model1': conv1, 'converged_model2': conv2,
            'iter_model1': it1, 'iter_model2': it2,
            'pred_up_rate_model1': pred1[in_eval].mean(),
            'pred_up_rate_model2': pred2[in_eval].mean(),
        })

        pids = s['playerID'].values
        names = s['name'].values
        for k in range(len(s)):
            rows.append({'group': group, 'metric': metric, 'playerID': pids[k], 'name': names[k],
                         'y2025': lv[k], 'change2025': ch[k],
                         'p_up_model1': p1[k], 'p_up_model2': p2[k],
                         'pred_model1': pred1[k], 'pred_model2': pred2[k], 'label2026': label26[k]})

        coefs.append({'group': group, 'metric': metric, 'model': 'model1', 'term': 'intercept', 'beta': b1[0]})
        coefs.append({'group': group, 'metric': metric, 'model': 'model1', 'term': 'level', 'beta': b1[1]})
        coefs.append({'group': group, 'metric': metric, 'model': 'model2', 'term': 'intercept', 'beta': b2[0]})
        coefs.append({'group': group, 'metric': metric, 'model': 'model2', 'term': 'level', 'beta': b2[1]})
        coefs.append({'group': group, 'metric': metric, 'model': 'model2', 'term': 'change', 'beta': b2[2]})

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT_DIR, 'logistic_results.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'logistic_predictions.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(coefs).to_csv(os.path.join(OUT_DIR, 'logistic_coefficients.csv'), index=False, encoding='utf-8-sig')

    show = res[['group', 'metric', 'n_train', 'n_test', 'up_train', 'up_test', 'majority_class',
                'acc_baseline', 'acc_model1', 'acc_model2', 'iter_model1', 'iter_model2']]
    print(show.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
