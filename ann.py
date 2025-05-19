# correct_approach_transfer.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

# 디버그 모드 설정
DEBUG = True

def debug_print(message):
    """디버그 메시지 출력"""
    if DEBUG:
        print(f"[DEBUG] {message}")
        sys.stdout.flush()  # 버퍼 즉시 비우기

class AirconDataset(Dataset):
    """에어컨 데이터셋 클래스"""
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class ImprovedANN(nn.Module):
    """개선된 신경망 모델"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 모델 구조
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def process_aircon_pair(df, id1, id2):
    """
    에어컨 쌍 처리 - 모든 쌍에서 동일한 변수명 패턴 사용
    id1: 첫 번째 에어컨 ID (항상 홀수)
    id2: 두 번째 에어컨 ID (항상 짝수)
    """
    debug_print(f"에어컨 쌍 처리: ID{id1}-ID{id2}")
    
    # 각 에어컨 ID에 해당하는 데이터 추출
    df1 = df[df['Auto Id'] == id1].reset_index(drop=True)
    df2 = df[df['Auto Id'] == id2].reset_index(drop=True)
    
    debug_print(f"ID{id1} 데이터 크기: {df1.shape}, ID{id2} 데이터 크기: {df2.shape}")
    
    # 5분 단위 샘플링
    df1 = df1.iloc[::60].reset_index(drop=True)
    df2 = df2.iloc[::60].reset_index(drop=True)
    
    # 변수명 변경 - 모든 쌍에서 동일한 패턴 사용
    # 첫 번째 에어컨은 항상 1, 두 번째 에어컨은 항상 2
    df1 = df1.rename(columns={
        'Tcon': 'tcon1', 
        'Tpip_in': 'Tpip_in1', 
        'Tpip_out': 'Tpip_out1',
        'Tbdy': 'Tbdy1', 
        'Tid': 'Tid1', 
        'Power': 'Power1'
    })
    
    df2 = df2.rename(columns={
        'Tcon': 'tcon2', 
        'Tpip_in': 'Tpip_in2', 
        'Tpip_out': 'Tpip_out2',
        'Tbdy': 'Tbdy2', 
        'Tid': 'Tid2', 
        'Power': 'Power2'
    })
    
    # 데이터 병합을 위한 필요 컬럼 선택
    columns1 = ['Time', 'tcon1', 'Tpip_in1', 'Tpip_out1', 'Tbdy1', 'Tid1', 'Power1', 'Tod']
    columns2 = ['Time', 'tcon2', 'Tpip_in2', 'Tpip_out2', 'Tbdy2', 'Tid2', 'Power2']
    
    # 존재하는 컬럼만 선택
    columns1 = [col for col in columns1 if col in df1.columns]
    columns2 = [col for col in columns2 if col in df2.columns]
    
    # 두 데이터프레임 병합
    merged = pd.merge(df1[columns1], df2[columns2], on='Time', how='inner')
    debug_print(f"병합 후 데이터 크기: {merged.shape}")
    
    # on/off 변수 추가
    merged['on_off1'] = merged['tcon1'].apply(lambda x: 1 if x != 0 and not pd.isna(x) else 0)
    merged['on_off2'] = merged['tcon2'].apply(lambda x: 1 if x != 0 and not pd.isna(x) else 0)
    
    # NaN 처리
    merged['tcon1'] = merged['tcon1'].fillna(0)
    merged['tcon2'] = merged['tcon2'].fillna(0)
    
    return merged

def load_data(file_path):
    """데이터 로드"""
    debug_print(f"파일 로드 중: {file_path}")
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    # 기본 전처리
    df.columns = df.columns.str.strip()
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df = df.sort_values(['Time', 'Auto Id'])
    
    debug_print(f"로드된 데이터 크기: {df.shape}")
    return df

def prepare_model_data(df):
    """모델 학습을 위한 데이터 준비"""
    debug_print("모델 데이터 준비 시작")
    
    # 타깃 변수 설정
    target_cols = ['Tpip_in1', 'Tpip_out1', 'Tbdy1', 'Tid1', 'Power1',
                   'Tpip_in2', 'Tpip_out2', 'Tbdy2', 'Tid2', 'Power2']
    
    debug_print(f"타깃 변수: {target_cols}")
    
    # 다음 시점 타깃 변수 생성
    for col in target_cols:
        df[f'{col}_next'] = df[col].shift(-1)
    
    # 시차 특성 추가 (t-1 시점)
    for col in target_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    
    # NaN 제거
    orig_len = len(df)
    df = df.dropna().reset_index(drop=True)
    removed = orig_len - len(df)
    debug_print(f"NaN 제거 후 {removed}행 제거됨. 남은 데이터: {len(df)}행")
    
    # 입력 및 출력 변수 정의
    input_cols = ['on_off1', 'tcon1', 'Tpip_in1', 'Tpip_out1', 'Tbdy1', 'Tid1', 'Power1',
                  'on_off2', 'tcon2', 'Tpip_in2', 'Tpip_out2', 'Tbdy2', 'Tid2', 'Power2', 'Tod']
    
    # 시차 특성 추가
    for col in target_cols:
        input_cols.append(f'{col}_lag1')
    
    output_cols = [f'{col}_next' for col in target_cols]
    
    debug_print(f"입력 변수 수: {len(input_cols)}, 출력 변수 수: {len(output_cols)}")
    
    # 모든 컬럼이 있는지 확인
    for col in input_cols + output_cols:
        if col not in df.columns:
            debug_print(f"경고: 컬럼이 데이터프레임에 없음: {col}")
    
    return df, input_cols, output_cols

def plot_time_series(true_values, pred_values, output_cols, pair_name, output_dir):
    """시계열 실제값과 예측값 시각화"""
    # 샘플 인덱스 (전체 데이터 또는 최대 100개만)
    sample_size = min(100, true_values.shape[0])
    if sample_size < true_values.shape[0]:
        sample_indices = np.random.choice(true_values.shape[0], sample_size, replace=False)
        sample_indices = np.sort(sample_indices)  # 시간 순서대로 정렬
    else:
        sample_indices = np.arange(sample_size)
    
    # 변수 타입별 그룹화 (온도 변수와 전력 변수)
    temp_vars = []
    power_vars = []
    
    for i, col in enumerate(output_cols):
        col_name = col.replace('_next', '')
        if 'Power' in col_name:
            power_vars.append((i, col_name))
        else:
            temp_vars.append((i, col_name))
    
    # 온도 변수 시각화
    if temp_vars:
        plt.figure(figsize=(14, 10))
        
        for i, (var_idx, var_name) in enumerate(temp_vars):
            plt.subplot(3, 3, i+1 if i < 9 else 9)  # 최대 9개까지 표시
            
            plt.plot(sample_indices, true_values[sample_indices, var_idx], 'b-', label='Actual')
            plt.plot(sample_indices, pred_values[sample_indices, var_idx], 'r--', label='Predicted')
            
            plt.title(f'{var_name} - {pair_name}')
            plt.xlabel('Sample Index')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if i >= 8:  # 9개 이상이면 나머지는 표시하지 않음
                break
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'time_series_temperature_{pair_name.replace(" ", "_")}.png'))
        plt.close()
    
    # 전력 변수 시각화
    if power_vars:
        plt.figure(figsize=(14, 6))
        
        for i, (var_idx, var_name) in enumerate(power_vars):
            plt.subplot(1, 2, i+1 if i < 2 else 2)  # 최대 2개까지 표시
            
            plt.plot(sample_indices, true_values[sample_indices, var_idx], 'b-', label='Actual')
            plt.plot(sample_indices, pred_values[sample_indices, var_idx], 'r--', label='Predicted')
            
            plt.title(f'{var_name} - {pair_name}')
            plt.xlabel('Sample Index')
            plt.ylabel('Power (W)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if i >= 1:  # 2개 이상이면 나머지는 표시하지 않음
                break
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'time_series_power_{pair_name.replace(" ", "_")}.png'))
        plt.close()

def train_model(df, input_cols, output_cols, epochs=100, patience=15):
    """모델 학습"""
    debug_print("모델 학습 시작")
    
    # 데이터 분할 및 스케일링
    X = df[input_cols].values
    Y = df[output_cols].values
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # 스케일러 생성 
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    
    # 학습 데이터로 스케일러 학습
    X_train_scaled = x_scaler.fit_transform(X_train)
    Y_train_scaled = y_scaler.fit_transform(Y_train)
    
    # 검증 데이터 스케일링
    X_val_scaled = x_scaler.transform(X_val)
    Y_val_scaled = y_scaler.transform(Y_val)
    
    debug_print(f"데이터 스케일링 완료. 학습 데이터 크기: {X_train_scaled.shape}, 검증 데이터 크기: {X_val_scaled.shape}")
    
    # 데이터 로더 생성
    train_loader = DataLoader(AirconDataset(X_train_scaled, Y_train_scaled), batch_size=32, shuffle=True)
    val_loader = DataLoader(AirconDataset(X_val_scaled, Y_val_scaled), batch_size=32)
    
    # 모델 생성
    input_dim = X_train_scaled.shape[1]
    output_dim = Y_train_scaled.shape[1]
    model = ImprovedANN(input_dim, output_dim)
    
    debug_print(f"모델 생성 완료. 입력 차원: {input_dim}, 출력 차원: {output_dim}")
    
    # 학습 설정
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 학습 결과 저장
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 학습 루프
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증 모드
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        
        # 손실값 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 조기 종료 로직
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 로그 출력
        debug_print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 조기 종료
        if patience_counter >= patience:
            debug_print(f"Early stopping after {epoch+1} epochs!")
            break
    
    # 최적 모델 상태로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, x_scaler, y_scaler, train_losses, val_losses, X_val_scaled, Y_val_scaled

def evaluate_model(model, X_val_scaled, Y_val_scaled, y_scaler, output_cols, pair_name="", output_dir=None):
    """모델 평가"""
    debug_print("모델 평가 중...")
    
    # 검증 데이터에 대한 예측
    model.eval()
    with torch.no_grad():
        val_predictions = model(torch.tensor(X_val_scaled, dtype=torch.float32)).numpy()
    
    # 원래 스케일로 변환
    val_pred_original = y_scaler.inverse_transform(val_predictions)
    val_true_original = y_scaler.inverse_transform(Y_val_scaled)
    
    # 성능 평가
    performance_data = []
    for i, col in enumerate(output_cols):
        rmse = np.sqrt(mean_squared_error(val_true_original[:, i], val_pred_original[:, i]))
        r2 = r2_score(val_true_original[:, i], val_pred_original[:, i])
        performance_data.append({
            'Target': col.replace('_next', ''),
            'RMSE': rmse,
            'R²': r2
        })
    
    performance_df = pd.DataFrame(performance_data)
    debug_print("\n모델 성능 평가:")
    debug_print(performance_df)
    
    # 시계열 시각화 추가
    if output_dir:
        plot_time_series(val_true_original, val_pred_original, output_cols, pair_name, output_dir)
    
    return performance_df, val_pred_original, val_true_original

def apply_model_to_pair(model, x_scaler, y_scaler, df, input_cols, output_cols, pair_name="", output_dir=None):
    """학습된 모델을 에어컨 쌍에 적용"""
    # 입출력 변수 존재 확인
    for col in input_cols + output_cols:
        if col not in df.columns:
            raise ValueError(f"컬럼이 데이터프레임에 없음: {col}")
    
    # 데이터 준비
    X = df[input_cols].values
    Y = df[output_cols].values
    
    # 스케일링
    X_scaled = x_scaler.transform(X)
    Y_scaled = y_scaler.transform(Y)
    
    # 예측
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    
    # 원래 스케일로 변환
    pred_original = y_scaler.inverse_transform(predictions)
    true_original = y_scaler.inverse_transform(Y_scaled)
    
    # 성능 평가
    performance_data = []
    for i, col in enumerate(output_cols):
        rmse = np.sqrt(mean_squared_error(true_original[:, i], pred_original[:, i]))
        r2 = r2_score(true_original[:, i], pred_original[:, i])
        performance_data.append({
            'Target': col.replace('_next', ''),
            'RMSE': rmse,
            'R²': r2
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # 시계열 시각화 추가
    if output_dir:
        plot_time_series(true_original, pred_original, output_cols, pair_name, output_dir)
    
    return performance_df, pred_original, true_original

def plot_learning_curves(train_losses, val_losses, save_path=None):
    """학습 곡선 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_performance_comparison(performance_dfs, pair_names, output_dir):
    """성능 비교 시각화"""
    # 각 타겟 변수별로 성능 비교
    for metric in ['RMSE', 'R²']:
        plt.figure(figsize=(14, 8))
        
        # 모든 쌍의 데이터 추출
        all_targets = []
        for df in performance_dfs:
            all_targets.extend(df['Target'].tolist())
        unique_targets = sorted(list(set(all_targets)))
        
        # 바 차트 데이터 준비
        bar_width = 0.8 / len(pair_names)
        indices = np.arange(len(unique_targets))
        
        for i, (df, name) in enumerate(zip(performance_dfs, pair_names)):
            values = []
            for target in unique_targets:
                if target in df['Target'].values:
                    values.append(df[df['Target'] == target][metric].values[0])
                else:
                    values.append(0)  # 해당 타겟이 없는 경우
            
            offset = i * bar_width - (len(pair_names) - 1) * bar_width / 2
            plt.bar(indices + offset, values, bar_width, label=name)
        
        plt.xlabel('Target Variable')
        plt.ylabel(metric)
        plt.title(f'{metric} Comparison Across Aircon Pairs')
        plt.xticks(indices, unique_targets, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
        plt.close()

def run_transfer_experiment():
    """전이 실험 실행"""
    # 설정
    file_path = '경남_사무실/LOG_SMARTCARE_20240904.csv'
    output_dir = 'correct_approach_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("===== 개선된 에어컨 모델 전이 실험 시작 =====")
    
    # 1. 데이터 로드
    df = load_data(file_path)
    
    # 2. 에어컨 쌍 처리
    aircon_pairs = [(1, 2), (3, 4), (5, 6)]
    pair_names = ['Pair 1-2', 'Pair 3-4', 'Pair 5-6']
    processed_dfs = []
    
    for pair in aircon_pairs:
        pair_df = process_aircon_pair(df, pair[0], pair[1])
        processed_dfs.append(pair_df)
    
    # 3. 모델 학습 데이터 준비 (첫 번째 쌍으로 학습)
    train_df, input_cols, output_cols = prepare_model_data(processed_dfs[0])
    
    # 4. 모델 학습
    model, x_scaler, y_scaler, train_losses, val_losses, X_val_scaled, Y_val_scaled = train_model(
        train_df, input_cols, output_cols
    )
    
    # 5. 학습 곡선 시각화
    plot_learning_curves(train_losses, val_losses, os.path.join(output_dir, 'learning_curves.png'))
    
    # 6. 첫 번째 쌍에 대한 모델 평가
    performance_df_pair1, _, _ = evaluate_model(
        model, X_val_scaled, Y_val_scaled, y_scaler, output_cols, 
        pair_name=pair_names[0], output_dir=output_dir
    )
    performance_df_pair1.to_csv(os.path.join(output_dir, 'performance_pair1-2.csv'), index=False)
    
    # 7. 다른 쌍에 대한 데이터 준비 및 모델 적용
    performance_dfs = [performance_df_pair1]
    
    for i in range(1, len(processed_dfs)):
        pair_df, pair_input_cols, pair_output_cols = prepare_model_data(processed_dfs[i])
        
        try:
            # 모델 적용 및 성능 평가
            pair_performance, pair_pred, pair_true = apply_model_to_pair(
                model, x_scaler, y_scaler, pair_df, input_cols, output_cols,
                pair_name=pair_names[i], output_dir=output_dir
            )
            
            # 결과 저장
            pair_performance.to_csv(os.path.join(output_dir, f'performance_{aircon_pairs[i][0]}-{aircon_pairs[i][1]}.csv'), index=False)
            performance_dfs.append(pair_performance)
            
            debug_print(f"{pair_names[i]}에 모델 적용 완료")
            
        except Exception as e:
            debug_print(f"{pair_names[i]}에 모델 적용 중 오류 발생: {e}")
            # 임시 데이터 생성 (비교 차트를 위해)
            performance_dfs.append(pd.DataFrame({'Target': performance_df_pair1['Target'], 'RMSE': 0, 'R²': 0}))
    
    # 8. 성능 비교 시각화
    plot_performance_comparison(performance_dfs, pair_names, output_dir)
    
    # 9. 모델 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'x_scaler': x_scaler,
        'y_scaler': y_scaler,
        'input_cols': input_cols,
        'output_cols': output_cols
    }, os.path.join(output_dir, 'aircon_model.pth'))
    
    print("\n===== 실험 결과 요약 =====")
    for i, df in enumerate(performance_dfs):
        print(f"\n{pair_names[i]} 성능:")
        print(df)
    
    print(f"\n===== 실험 완료. 결과는 '{output_dir}' 디렉토리에 저장되었습니다. =====")
    
    # 시계열 시각화 파일 목록 출력
    time_series_files = [f for f in os.listdir(output_dir) if f.startswith('time_series_')]
    if time_series_files:
        print("\n생성된 시계열 시각화 파일:")
        for file in time_series_files:
            print(f"- {file}")

if __name__ == "__main__":
    # sklearn 모듈 임포트
    from sklearn.metrics import mean_squared_error, r2_score
    
    # 전이 실험 실행
    run_transfer_experiment()