#!/usr/bin/env python3
# tau_by_space_dynamic_matrix_fixed_weight.py
# Usage:
#   python tau_by_space_dynamic_matrix_fixed_weight.py LOG_SMARTCARE_20240913.csv "[[1,2,3],[4,5],[6]]" --win 15
#
# Dependencies:
#   pip install pandas numpy scipy matplotlib

import ast
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="로그 CSV 파일 경로")
    parser.add_argument(
        "spaces",
        help='공간별 Auto-Id 리스트 문자열, 예: "[[1,2,3],[4,5],[6]]"'
    )
    parser.add_argument(
        "--win", type=int, default=15,
        help="슬라이딩 윈도우 크기(분), 기본값 15"
    )
    return parser.parse_args()

def pivot_space(df, auto_ids):
    df_pivot = {}
    for col in ["Tid", "Tcon"]:
        df_pivot[col] = df.pivot(index="DateTime", columns="Auto Id", values=col)
        df_pivot[col].columns = [f"{col}_{aid}" for aid in df_pivot[col].columns]
    df_pivot = pd.concat(df_pivot.values(), axis=1)
    df_pivot["Tod"] = df.groupby("DateTime")["Tod"].first()
    df_pivot = df_pivot.sort_index()
    for aid in auto_ids:
        for col in ["Tid", "Tcon"]:
            cname = f"{col}_{aid}"
            if cname not in df_pivot.columns:
                df_pivot[cname] = np.nan
    df_pivot = df_pivot[[f"Tid_{aid}" for aid in auto_ids] + [f"Tcon_{aid}" for aid in auto_ids] + ["Tod"]]
    return df_pivot

def estimate_weight_once(W, samples, auto_ids, on_mask):
    n = len(auto_ids)
    updated = False
    for i in range(n):
        for j in range(n):
            if on_mask[i, j] and W[i, j] == 0:
                rows = []
                targets = []
                for k in range(len(samples)):
                    Tcon_vec = np.array([samples.iloc[k][f"Tcon_{aid}"] for aid in auto_ids])
                    if Tcon_vec[j] == 0:
                        continue
                    mask = Tcon_vec != 0
                    weights_row = W[i, :].copy()
                    weights_row[~mask] = 0
                    weights_row[j] = 1  # 추정 대상만 1로 두고 나머지는 기존값
                    num = np.sum(weights_row * Tcon_vec)
                    den = np.sum(weights_row)
                    if den == 0:
                        continue
                    Ttarget = num / den
                    Tid = samples.iloc[k][f"Tid_{auto_ids[i]}"]
                    rows.append([Tcon_vec[j]])
                    targets.append(Tid - np.sum(weights_row * Tcon_vec) + weights_row[j]*Tcon_vec[j])
                if rows and targets:
                    A = np.array(rows)
                    b = np.array(targets)
                    res = lsq_linear(A, b, bounds=(0, 10))
                    W[i, j] = res.x[0]
                    updated = True
    return W, updated

def compute_Ttarget(W, Tcon, Tod):
    Ttargets = []
    for i in range(W.shape[0]):
        weights = W[i, :]
        mask = Tcon != 0
        if mask.any() and np.sum(weights[mask]) > 0:
            num = np.sum(weights[mask] * Tcon[mask])
            den = np.sum(weights[mask])
            Ttarget = num / den
        else:
            Ttarget = Tod
        Ttargets.append(Ttarget)
    return np.array(Ttargets)

def estimate_tau(samples, Ttargets, auto_ids):
    n = len(auto_ids)
    tau_vec = np.full(n, 30.0)
    for i, aid in enumerate(auto_ids):
        Tid = samples[f"Tid_{aid}"].values
        Ttar = Ttargets[:, i]
        if len(Tid) < 2:
            continue
        def obj(tau):
            # if tau <= 0:
            #     return 1e6
            pred = Ttar[:-1] + (Tid[:-1] - Ttar[:-1]) * np.exp(-1.0 / tau)
            return np.mean((Tid[1:] - pred) ** 2)
        try:
            from scipy.optimize import minimize
            res = minimize(obj, x0=[30], bounds=[(-1e4, 1e4)])
            if res.success and np.isfinite(res.x[0]) and res.x[0] > 0:
                tau_vec[i] = res.x[0]
        except Exception:
            pass
    return tau_vec

def main():
    args = parse_args()
    try:
        spaces = ast.literal_eval(args.spaces)
        assert all(isinstance(lst, (list, tuple)) for lst in spaces)
    except (ValueError, AssertionError):
        raise SystemExit("⛔ spaces 인자를 올바른 리스트 형식으로 입력하세요.")
    df = pd.read_csv(args.csv)

    if "DateTime" not in df.columns:
        # base = pd.to_datetime("2025-07-09")
        # df["DateTime"] = pd.to_datetime(base.strftime("%Y-%m-%d") + " " + df["Time"])
        df["DateTime"] = pd.to_datetime(df["Time"])

    for col in ["Tid", "Tod", "Tcon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_5min_list = []
    for aid in df["Auto Id"].unique():
        df_sub = df[df["Auto Id"] == aid].copy()
        df_sub = df_sub.set_index("DateTime")
        df_sub_5min = df_sub.resample("5min").mean(numeric_only=True)
        df_sub_5min["Auto Id"] = aid
        df_sub_5min = df_sub_5min.reset_index()
        df_5min_list.append(df_sub_5min)
    df_5min = pd.concat(df_5min_list, ignore_index=True)

    win = args.win# // 0.5

    pred_records = []
    tau_records = []  # tau 기록용

    for space_idx, auto_ids in enumerate(spaces):
        print(f"\n=== Space {space_idx+1} (Auto Ids: {auto_ids}) ===")
        df_space = df_5min[df_5min["Auto Id"].isin(auto_ids)].copy()
        df_space["DateTime"] = pd.to_datetime(df_space["DateTime"])
        df_pivot = pivot_space(df_space, auto_ids)
        n = len(auto_ids)
        W = np.zeros((n, n))  # W 행렬 초기화
        W_fixed = np.zeros((n, n), dtype=bool)  # 고정 여부
        for i in range(len(df_pivot) - win):
            samples = df_pivot.iloc[i:i+win+1]
            Tcon_mat = samples[[f"Tcon_{aid}" for aid in auto_ids]].values
            Tod_vec = samples["Tod"].values
            on_mask = (Tcon_mat != 0)
            # 최초 ON된 실내기 조합에 대해만 W 업데이트
            for ii in range(n):
                for jj in range(n):
                    if not W_fixed[ii, jj] and np.any(on_mask[:, jj]):
                        rows = []
                        targets = []
                        for k in range(len(samples)):
                            Tcon_vec = Tcon_mat[k]
                            if Tcon_vec[jj] == 0:
                                continue
                            mask = Tcon_vec != 0
                            weights_row = W[ii, :].copy()
                            weights_row[~mask] = 0
                            weights_row[jj] = 1
                            num = np.sum(weights_row * Tcon_vec)
                            den = np.sum(weights_row)
                            if den == 0:
                                continue
                            Ttarget = num / den
                            Tid = samples.iloc[k][f"Tid_{auto_ids[ii]}"]
                            rows.append([Tcon_vec[jj]])
                            targets.append(Tid - np.sum(weights_row * Tcon_vec) + weights_row[jj]*Tcon_vec[jj])
                        if rows and targets:
                            A = np.array(rows)
                            b = np.array(targets)
                            res = lsq_linear(A, b, bounds=(0, 1))
                            W[ii, jj] = res.x[0]
                            W_fixed[ii, jj] = True
            # 각 시점별로 Ttarget 계산
            Ttargets_mat = []
            for k in range(len(samples)):
                Tcon = Tcon_mat[k]
                Tod = Tod_vec[k]
                Ttargets = compute_Ttarget(W, Tcon, Tod)
                Ttargets_mat.append(Ttargets)
            Ttargets_mat = np.array(Ttargets_mat)
            tau_vec = estimate_tau(samples, Ttargets_mat, auto_ids)
            # tau 값 기록
            timestamp = samples.index[-1]
            for idx, aid in enumerate(auto_ids):
                tau = tau_vec[idx] if np.isfinite(tau_vec[idx]) and tau_vec[idx] > 0 else 30.0
                tau_records.append({
                    "Auto_Id": aid,
                    "timestamp": timestamp,
                    "tau": tau
                })
            # 예측 및 기록 (윈도우 마지막 2개 시점)
            for idx, aid in enumerate(auto_ids):
                if len(samples) < 2:
                    continue
                cur_idx = -2
                nxt_idx = -1
                Tid_now = samples.iloc[cur_idx][f"Tid_{aid}"]
                Tid_next_true = samples.iloc[nxt_idx][f"Tid_{aid}"]
                Ttarget = Ttargets_mat[cur_idx, idx]
                tau = tau_vec[idx] if np.isfinite(tau_vec[idx]) and tau_vec[idx] > 0 else 30.0
                Tid_pred = Ttarget + (Tid_now - Ttarget) * np.exp(-1.0 / tau)
                pred_records.append({
                    "Auto_Id": aid,
                    "timestamp": samples.index[nxt_idx],
                    "Tid_true": Tid_next_true,
                    "Tid_pred": Tid_pred,
                    "Ttarget": Ttargets_mat[nxt_idx, idx]
                })
        np.savetxt(f"space{space_idx+1}_W_matrix.csv", W, delimiter=",")
    pred_df = pd.DataFrame(pred_records)
    tau_df = pd.DataFrame(tau_records)
    if not pred_df.empty:
        # Auto Id별 예측 오차(%) 계산 및 출력
        print("\n=== Auto Id별 예측 오차(%) ===")
        for aid in pred_df["Auto_Id"].unique():
            grp = pred_df[pred_df["Auto_Id"] == aid]
            mae = np.mean(np.abs(grp["Tid_true"] - grp["Tid_pred"]))
            true_mean = np.mean(np.abs(grp["Tid_true"]))
            if true_mean == 0:
                error_pct = np.nan
            else:
                error_pct = 100 * mae / true_mean
            print(f"Auto_Id {aid}: 평균 절대 오차(MAE) = {mae:.3f}, 상대 오차 = {error_pct:.2f}%")

    if not pred_df.empty:
        for aid in pred_df["Auto_Id"].unique():
            grp = pred_df[pred_df["Auto_Id"] == aid]
            plt.figure(figsize=(15, 6))
            plt.plot(grp["timestamp"], grp["Tid_true"], label="True", color="b")
            plt.plot(grp["timestamp"], grp["Tid_pred"], label="Predict", color="r", linestyle="--")
            # plt.plot(grp["timestamp"], grp["Ttarget"], label="Ttarget", color="g", linestyle=":")
            plt.title(f"Auto_Id {aid}: True vs Predict Tid (24 hours)")
            plt.xlabel("Time")
            plt.ylabel("Tid (°C)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"tid_true_vs_pred_autoid_{aid}.png", dpi=200)
            plt.close()

    # tau 값 csv 저장
    if not tau_df.empty:
        tau_df.to_csv("tau_values_by_auto_id.csv", index=False)

    # tau 값 그래프 저장
    if not tau_df.empty:
        for aid in tau_df["Auto_Id"].unique():
            grp = tau_df[tau_df["Auto_Id"] == aid]
            plt.figure(figsize=(10,5))
            plt.plot(grp["timestamp"], grp["tau"])
            plt.title(f"Tau values over time for Auto_Id {aid}")
            plt.xlabel("Time")
            plt.ylabel("Tau")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"tau_values_autoid_{aid}.png", dpi=200)
            plt.close()

if __name__ == "__main__":
    main()
