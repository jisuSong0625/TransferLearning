from datetime import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error

# 1. 데이터 로딩 및 시간 처리
df = pd.read_csv('../경남_사무실/LOG_SMARTCARE_20240912.csv')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
df.set_index('Time', inplace=True)

# 2. 분석할 에어컨 Auto Id 지정
target_ids = [1, 2, 3, 4]  # 원하는 에어컨 번호로 수정

for auto_id in target_ids:
    # 3. 해당 에어컨 데이터만 추출
    df_ac = df[df['Auto Id'] == auto_id].copy()
    numeric_cols = df_ac.select_dtypes(include=[np.number]).columns
    sampled_df = df_ac[numeric_cols].resample('5min').mean()
    sampled_df = sampled_df.dropna(subset=['Tid', 'Tcon', 'Tod']).copy()

    # 4. 예측값, tau 저장 컬럼
    sampled_df['Tid_pred'] = np.nan
    sampled_df['tau'] = np.nan

    window_size = 2  # 1시간 = 12개(5분 간격)
    tau_min, tau_max = 60, 7200  # 1분~2시간(초)

    for i in range(window_size, len(sampled_df)):
        window = sampled_df.iloc[i-window_size:i]
        t = (window.index - window.index[0]).total_seconds()
        T_true = window['Tid'].values
        T0 = T_true[0]
        Tcon_now = sampled_df['Tcon'].iloc[i]
        Tod_now = sampled_df['Tod'].iloc[i]
        tau_candidates = np.linspace(tau_min, tau_max, 30)
        best_tau = tau_candidates[0]
        best_rmse = np.inf
        if Tcon_now != 0:
            for tau in tau_candidates:
                pred = Tcon_now + (T0 - Tcon_now) * np.exp(-t / tau)
                score = np.sqrt(np.mean((T_true - pred) ** 2))
                if score < best_rmse:
                    best_rmse = score
                    best_tau = tau
            dt = (sampled_df.index[i] - window.index[-1]).total_seconds()
            prev = T_true[-1]
            pred_val = Tcon_now + (prev - Tcon_now) * np.exp(-dt / best_tau)
        else:
            for tau in tau_candidates:
                pred = Tod_now + (T0 - Tod_now) * np.exp(-t / tau)
                score = np.sqrt(np.mean((T_true - pred) ** 2))
                if score < best_rmse:
                    best_rmse = score
                    best_tau = tau
            dt = (sampled_df.index[i] - window.index[-1]).total_seconds()
            prev = T_true[-1]
            pred_val = Tod_now + (prev - Tod_now) * np.exp(-dt / best_tau)
        sampled_df.iloc[i, sampled_df.columns.get_loc('Tid_pred')] = pred_val
        sampled_df.iloc[i, sampled_df.columns.get_loc('tau')] = best_tau

    # 5.0. 24시간 예측 그래프 저장
    plt.figure(figsize=(15, 6))
    plt.plot(sampled_df.index, sampled_df['Tid'], 'b-', label='True Tid')
    plt.plot(sampled_df.index, sampled_df['Tid_pred'], 'r--', label='1-hour Sliding Exp Prediction', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Temperature (Tid)')
    plt.title(f'Auto Id {auto_id} - 24-hour Tid with 1-hour Sliding Window Exponential Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.savefig(f'AutoId{auto_id}_tid_sliding_1h_exp_pred.png', dpi=150)
    plt.close()

    # 5.1. 구간 그래프 저장
    start_time = sampled_df.index[0] + pd.Timedelta(hours=7)
    end_time = start_time + pd.Timedelta(hours=1)
    first_3h_df = sampled_df[(sampled_df.index >= start_time) & (sampled_df.index < end_time)]
    
    plt.figure(figsize=(15, 6))
    plt.plot(first_3h_df.index, first_3h_df['Tid'], 'b-', label='True Tid')
    plt.plot(first_3h_df.index, first_3h_df['Tid_pred'], 'r--', label='1-hour Sliding Exp Prediction', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Temperature (Tid)')
    plt.title(f'Auto Id {auto_id} - Part Hours Tid with 1-hour Sliding Window Exponential Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.savefig(f'AutoId{auto_id}_tid_part_hours_exp_pred.png', dpi=150)
    plt.close()

    # 6. RMSE 및 일치율(%) 계산
    valid_idx = sampled_df['Tid_pred'].notna() & sampled_df['Tid'].notna()
    if valid_idx.sum() > 0:
        rms_error = np.sqrt(mean_squared_error(sampled_df.loc[valid_idx, 'Tid'], sampled_df.loc[valid_idx, 'Tid_pred']))
        tid_range = sampled_df['Tid'].max() - sampled_df['Tid'].min()
        match_percent = 100 * (1 - rms_error / tid_range) if tid_range > 0 else 0
        print(f"[Auto Id {auto_id}] Overall RMSE: {rms_error:.4f}")
        print(f"[Auto Id {auto_id}] 일치율: {match_percent:.2f}%")
    else:
        print(f"[Auto Id {auto_id}] 유효한 예측값이 없어 RMSE 계산 불가")

    # 7. tau 값 CSV 저장
    tau_df = sampled_df.loc[valid_idx, ['tau']].copy()
    tau_df['Time'] = tau_df.index.strftime('%H:%M:%S')
    tau_df = tau_df[['Time', 'tau']]
    tau_df.to_csv(f'AutoId{auto_id}_tau_values.csv', index=False)

    # 8. tau 시계열 그래프 저장
    plt.figure(figsize=(15, 6))
    plt.plot(tau_df['Time'], tau_df['tau'], 'g-', label='Tau over time')
    plt.xlabel('Time')
    plt.ylabel('Tau (seconds)')
    plt.title(f'Auto Id {auto_id} - Tau values over time (1-hour sliding window)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.savefig(f'AutoId{auto_id}_tau_values_plot.png', dpi=150)
    plt.close()
