import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import glob
import os

class HVACDataProcessor:
    """HVAC 로그 데이터 전처리 클래스 - Auto ID별 5분 간격 샘플링 수정 버전"""
    
    def __init__(self):
        self.state_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        
    def load_data(self, file_pattern: str) -> pd.DataFrame:
        """CSV 파일들을 올바르게 로드"""
        files = glob.glob(file_pattern)
        files.sort()
        
        all_sequences = []
        
        for file in files:
            print(f"Processing {file}...")
            try:
                df = pd.read_csv(file)
                # 시간 파싱
                try:
                    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')
                except:
                    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
                
                # NaT 값 제거
                df = df.dropna(subset=['Time'])
                
                if len(df) == 0:
                    print(f"  ⚠️ No valid data in {file}")
                    continue
                
                # 파일 내에서 시간순 정렬
                df = df.sort_values('Time').reset_index(drop=True)
                
                # Auto ID별로 5분 간격 리샘플링
                resampled_data = self.resample_by_auto_id(df)
                
                if len(resampled_data) > 0:
                    print(f"  ✅ {len(df)} → {len(resampled_data)} rows (Auto ID별 5분 간격)")
                    # 파일별 식별자 추가
                    resampled_data['file_source'] = file
                    all_sequences.append(resampled_data)
                else:
                    print(f"  ⚠️ No data after resampling in {file}")
                
            except Exception as e:
                print(f"  ❌ Error loading {file}: {e}")
                continue
        
        if not all_sequences:
            raise ValueError("No valid data found in any files")
        
        combined_df = pd.concat(all_sequences, ignore_index=True)
        
        print(f"✅ Total processed: {len(combined_df)} rows from {len(all_sequences)} files")
        return combined_df
    
    def resample_by_auto_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto ID별로 5분 간격 리샘플링"""
        print(f"🔄 Auto ID별 5분 간격 리샘플링 시작...")
        
        # Auto ID별로 그룹화
        auto_ids = df['Auto Id'].unique()
        print(f"   발견된 Auto ID: {sorted(auto_ids)}")
        
        resampled_groups = []
        
        for auto_id in auto_ids:
            auto_df = df[df['Auto Id'] == auto_id].copy()
            auto_df = auto_df.sort_values('Time').reset_index(drop=True)
            
            if len(auto_df) < 2:
                print(f"   Auto ID {auto_id}: 데이터 부족 ({len(auto_df)}개) - 건너뜀")
                continue
            
            # 시간 간격 분석
            time_diff = auto_df['Time'].diff().dropna()
            if len(time_diff) > 0:
                median_interval = time_diff.median().total_seconds()
                print(f"   Auto ID {auto_id}: {len(auto_df)}개 데이터, 중간값 간격 {median_interval:.0f}초")
            
            # 5분(300초) 간격으로 리샘플링
            target_interval = pd.Timedelta(seconds=300)  # 5분
            
            # 시작 시간부터 5분 간격으로 샘플링
            start_time = auto_df['Time'].iloc[0]
            end_time = auto_df['Time'].iloc[-1]
            
            # 5분 간격 시간 인덱스 생성
            time_index = pd.date_range(start=start_time, end=end_time, freq='5T')
            
            if len(time_index) < 2:
                print(f"   Auto ID {auto_id}: 시간 범위 부족 - 건너뜀")
                continue
            
            # 가장 가까운 시간의 데이터 선택 (nearest neighbor 방식)
            resampled_rows = []
            
            for target_time in time_index:
                # 목표 시간과 가장 가까운 데이터 찾기
                time_diffs = np.abs((auto_df['Time'] - target_time).dt.total_seconds())
                closest_idx = time_diffs.idxmin()
                
                # 5분 이내의 데이터만 사용 (너무 멀리 떨어진 데이터는 제외)
                if time_diffs.loc[closest_idx] <= 300:  # 5분 이내
                    resampled_rows.append(auto_df.loc[closest_idx])
            
            if len(resampled_rows) > 0:
                resampled_auto_df = pd.DataFrame(resampled_rows).reset_index(drop=True)
                # 중복 제거 (같은 원본 데이터가 여러 번 선택될 수 있음)
                resampled_auto_df = resampled_auto_df.drop_duplicates(subset='Time').reset_index(drop=True)
                resampled_groups.append(resampled_auto_df)
                print(f"   Auto ID {auto_id}: {len(auto_df)} → {len(resampled_auto_df)}개 (5분 간격)")
            else:
                print(f"   Auto ID {auto_id}: 리샘플링 후 데이터 없음")
        
        if len(resampled_groups) == 0:
            print("   ⚠️ 모든 Auto ID에서 리샘플링 실패")
            return pd.DataFrame()
        
        # 모든 Auto ID의 리샘플링된 데이터 결합
        result_df = pd.concat(resampled_groups, ignore_index=True)
        result_df = result_df.sort_values(['Time', 'Auto Id']).reset_index(drop=True)
        
        print(f"   ✅ 리샘플링 완료: 총 {len(result_df)}개 데이터")
        
        # Auto ID별 최종 데이터 개수 확인
        final_counts = result_df['Auto Id'].value_counts().sort_index()
        print(f"   📊 Auto ID별 최종 데이터:")
        for auto_id, count in final_counts.items():
            print(f"      Auto ID {auto_id}: {count}개")
        
        return result_df
    
    def extract_features_simplified(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """단순화된 특성 추출 - 페어링 없이 개별 처리"""
        
        print("🔍 단순화된 데이터 처리 중...")
        print(f"전체 데이터 크기: {df.shape}")
        print(f"Auto Id 범위: {df['Auto Id'].min()} ~ {df['Auto Id'].max()}")
        print(f"고유 Auto Id 개수: {df['Auto Id'].nunique()}")
        
        # 필요한 컬럼들 확인 - Power 제외
        required_state_cols = ['Tpip_in', 'Tpip_out', 'Tbdy', 'Tid', 'Tod']  # Power 제거
        required_action_cols = ['Tcon']
        
        available_state_cols = [col for col in required_state_cols if col in df.columns]
        available_action_cols = [col for col in required_action_cols if col in df.columns]
        
        print(f"사용 가능한 State 컬럼 (Power 제외): {available_state_cols}")
        print(f"사용 가능한 Action 컬럼: {available_action_cols}")
        print(f"📝 Power는 측정 주기가 달라 World Model에서 제외됨")
        
        if not available_state_cols or not available_action_cols:
            raise ValueError(f"필수 컬럼이 없습니다. State: {available_state_cols}, Action: {available_action_cols}")
        
        # 결측값 처리 - 일반적인 방식으로 단순화
        df_clean = df[available_state_cols + available_action_cols + ['file_source', 'Auto Id', 'Time']].copy()
        df_clean = df_clean.fillna(method='ffill').fillna(0)
        
        # State 특성 (개별 에어컨)
        states = df_clean[available_state_cols].values
        
        # Action 특성 생성
        tcon_values = df_clean[available_action_cols].values
        
        # on/off 특성 (Tcon이 0이 아니면 1)
        on_off = (tcon_values != 0).astype(int)
        
        # Ptarget (목표 압력) - 기본값
        ptarget = np.full((len(tcon_values), 1), 25.0)
        
        # Action 결합: [Tcon, on/off, Ptarget]
        actions = np.hstack([tcon_values, on_off, ptarget])
        
        # 파일 소스 정보 (Auto ID와 시간 정보 포함)
        file_sources = df_clean.apply(lambda row: f"{row['file_source']}_AutoID_{row['Auto Id']}", axis=1).values
        
        print(f"✅ 처리 완료:")
        print(f"   States shape: {states.shape}")
        print(f"   Actions shape: {actions.shape}")
        print(f"   File sources: {len(set(file_sources))}개 고유 시퀀스")
        
        return states, actions, file_sources
    
    def extract_features_paired(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """에어컨 쌍으로 처리하는 특성 추출 - 5분 간격 데이터 기반"""
        
        print("🔍 페어링 데이터 처리 중...")
        print(f"전체 데이터 크기: {df.shape}")
        
        # Auto Id 분석
        auto_ids = df['Auto Id'].unique()
        print(f"Auto Id 범위: {auto_ids.min()} ~ {auto_ids.max()}")
        print(f"고유 Auto Id: {sorted(auto_ids)}")
        
        # AC_Pair와 AC_Unit 계산
        df['AC_Pair'] = (df['Auto Id'] + 1) // 2
        df['AC_Unit'] = (df['Auto Id'] % 2) + 1
        
        pair_unit_counts = df.groupby(['AC_Pair', 'AC_Unit']).size().unstack(fill_value=0)
        print(f"AC_Pair별 Unit 분포:")
        print(pair_unit_counts)
        
        # 페어링 가능한 AC_Pair 찾기
        paired_acs = pair_unit_counts[(pair_unit_counts[1] > 0) & (pair_unit_counts[2] > 0)].index
        print(f"페어링 가능한 AC_Pair: {list(paired_acs)}")
        
        if len(paired_acs) == 0:
            print("⚠️ 페어링 가능한 AC가 없습니다. 단순화된 방식으로 전환합니다.")
            return self.extract_features_simplified(df)
        
        # 페어링된 데이터 생성 - 5분 간격 데이터 기반
        grouped_data = []
        file_sources = []
        
        for pair_id in paired_acs:
            pair_data = df[df['AC_Pair'] == pair_id].copy()
            
            # 시간별로 그룹화 (5분 간격으로 이미 리샘플링된 데이터)
            time_groups = pair_data.groupby('Time')
            
            pair_matches = 0
            for time, group in time_groups:
                group_1 = group[group['AC_Unit'] == 1]
                group_2 = group[group['AC_Unit'] == 2]
                
                if len(group_1) > 0 and len(group_2) > 0:
                    unit1 = group_1.iloc[0]
                    unit2 = group_2.iloc[0]
                    
                    # State 특성 추출 (안전한 방식)
                    def safe_get(row, col, default=0.0):
                        val = row.get(col, default)
                        if pd.isna(val) or val is None:
                            return float(default)
                        return float(val)
                    
                    state_features = [
                        safe_get(unit1, 'Tpip_in'), safe_get(unit2, 'Tpip_in'),
                        safe_get(unit1, 'Tpip_out'), safe_get(unit2, 'Tpip_out'),
                        safe_get(unit1, 'Tbdy'), safe_get(unit2, 'Tbdy'),
                        safe_get(unit1, 'Tid'), safe_get(unit2, 'Tid'),
                        safe_get(unit1, 'Tod')  # Tod는 공통, Power 제거
                    ]
                    
                    # Action 특성 추출
                    tcon1 = safe_get(unit1, 'Tcon')
                    tcon2 = safe_get(unit2, 'Tcon')
                    
                    action_features = [
                        tcon1, tcon2,
                        1 if tcon1 != 0 else 0,  # on/off1
                        1 if tcon2 != 0 else 0,  # on/off2
                        25.0  # Ptarget
                    ]
                    
                    grouped_data.append({
                        'time': time,
                        'pair_id': pair_id,
                        'states': state_features,
                        'actions': action_features
                    })
                    
                    file_sources.append(f"{unit1.get('file_source', 'unknown')}_Pair_{pair_id}")
                    pair_matches += 1
            
            print(f"  AC_Pair {pair_id}: {pair_matches}개 매칭된 시점 (5분 간격)")
        
        if not grouped_data:
            print("⚠️ 매칭된 데이터가 없습니다. 단순화된 방식으로 전환합니다.")
            return self.extract_features_simplified(df)
        
        states = np.array([item['states'] for item in grouped_data])
        actions = np.array([item['actions'] for item in grouped_data])
        file_sources = np.array(file_sources)
        
        print(f"✅ 페어링 완료: {len(states)}개 샘플 (5분 간격)")
        print(f"   States shape: {states.shape}")
        print(f"   Actions shape: {actions.shape}")
        
        return states, actions, file_sources
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """메인 특성 추출 함수 - 페어링 시도 후 실패시 단순화 방식"""
        try:
            return self.extract_features_paired(df)
        except Exception as e:
            print(f"⚠️ 페어링 처리 실패: {e}")
            print("단순화된 방식으로 처리합니다...")
            return self.extract_features_simplified(df)
    
    def create_sequences_per_auto_id(self, states: np.ndarray, actions: np.ndarray, 
                                    file_sources: np.ndarray, seq_length: int = 10, 
                                    prediction_horizon: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Auto ID별로 독립적인 시퀀스 생성 - 더 긴 예측 간격으로 수정"""
        
        # 정규화
        states_normalized = self.state_scaler.fit_transform(states)
        actions_normalized = self.action_scaler.fit_transform(actions)
        
        X_states, X_actions, y_states = [], [], []
        
        # Auto ID/파일별로 그룹화
        unique_sources = np.unique(file_sources)
        print(f"📁 Auto ID별 시퀀스 생성: {len(unique_sources)}개 고유 시퀀스")
        print(f"🎯 예측 간격: {prediction_horizon}스텝 ({prediction_horizon * 5}분 후 예측)")
        
        for source_name in unique_sources:
            # 안전한 소스명 처리
            try:
                source_display_name = str(source_name)
                if '\\' in source_display_name:
                    source_display_name = source_display_name.split('\\')[-1]
                elif '/' in source_display_name:
                    source_display_name = source_display_name.split('/')[-1]
            except:
                source_display_name = f"source_{hash(str(source_name)) % 1000}"
            
            source_mask = file_sources == source_name
            source_states = states_normalized[source_mask]
            source_actions = actions_normalized[source_mask]
            
            # 각 Auto ID/파일 내에서만 시퀀스 생성 (더 긴 예측 간격)
            source_sequences = 0
            if len(source_states) > seq_length + prediction_horizon:
                for i in range(len(source_states) - seq_length - prediction_horizon):
                    X_states.append(source_states[i:i+seq_length])
                    X_actions.append(source_actions[i:i+seq_length])
                    # prediction_horizon 스텝 후의 상태를 예측 (더 어려운 예측)
                    y_states.append(source_states[i+seq_length+prediction_horizon-1])
                    source_sequences += 1
            
            print(f"  {source_display_name}: {len(source_states)} rows → {source_sequences} sequences")
        
        if len(X_states) == 0:
            raise ValueError("시퀀스를 생성할 수 없습니다. 데이터가 부족합니다.")
        
        print(f"✅ 총 {len(X_states)}개 시퀀스 생성 (Auto ID별 독립적, {prediction_horizon * 5}분 후 예측)")
        
        return np.array(X_states), np.array(X_actions), np.array(y_states)

class HVACDataset(Dataset):
    """PyTorch Dataset 클래스"""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.targets[idx]

class SimpleSSMWorldModel(nn.Module):
    """단순화된 SSM World Model"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, 
                 latent_dim: int = 32):
        super(SimpleSSMWorldModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # State Space Model
        self.state_transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 시간적 가중치
        self.temporal_weight = nn.Parameter(torch.tensor(0.8))
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """순전파"""
        batch_size, seq_len, _ = states.shape
        
        # 최근 몇 개 시점만 사용
        use_seq_len = min(3, seq_len)
        recent_states = states[:, -use_seq_len:]
        recent_actions = actions[:, -use_seq_len:]
        
        # 각 시점 인코딩
        encoded_features = []
        for t in range(use_seq_len):
            combined = torch.cat([recent_states[:, t], recent_actions[:, t]], dim=-1)
            encoded = self.encoder(combined)
            encoded_features.append(encoded)
        
        # 가중 평균
        if use_seq_len == 1:
            final_encoded = encoded_features[0]
        else:
            weights = torch.softmax(torch.linspace(0.1, 1.0, use_seq_len).to(states.device), dim=0)
            final_encoded = sum(w * enc for w, enc in zip(weights, encoded_features))
        
        # State transition
        last_action = recent_actions[:, -1]
        transition_input = torch.cat([final_encoded, last_action], dim=-1)
        next_latent = self.state_transition(transition_input)
        
        # 시간적 연속성
        temporal_factor = torch.sigmoid(self.temporal_weight)
        last_state = recent_states[:, -1]
        
        # 디코딩
        decoded_change = self.decoder(next_latent)
        
        # 점진적 변화
        predicted_state = temporal_factor * last_state + (1 - temporal_factor) * decoded_change
        
        return predicted_state
    
    def predict_next_state(self, current_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """단일 스텝 예측"""
        with torch.no_grad():
            if len(current_state.shape) == 1:
                current_state = current_state.unsqueeze(0)
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
                
            state_seq = current_state.unsqueeze(1)
            action_seq = action.unsqueeze(1)
            
            prediction = self.forward(state_seq, action_seq)
            
            return prediction.squeeze(0) if prediction.shape[0] == 1 else prediction

class SSMTrainer:
    """SSM World Model 훈련 클래스"""
    
    def __init__(self, model: SimpleSSMWorldModel, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.7)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        
        for states, actions, targets in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(states, actions)
            loss = self.criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for states, actions, targets in val_loader:
                predictions = self.model(states, actions)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, patience: int = 10) -> Dict:
        """전체 훈련 프로세스"""
        best_val_loss = float('inf')
        patience_counter = 0
        min_improvement = 1e-4
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            improvement = best_val_loss - val_loss
            
            print(f"Epoch {epoch+1:2d}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                  f"LR: {current_lr:.1e}, Gap: {val_loss-train_loss:.4f}")
            
            if improvement > min_improvement:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_simple_ssm_model.pth')
                print(f"    ✓ New best! Improvement: {improvement:.4f}")
            else:
                patience_counter += 1
                print(f"    No improvement ({patience_counter}/{patience})")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            if val_loss - train_loss > 0.5:
                print(f"Overfitting detected! Gap: {val_loss - train_loss:.4f}")
                break
                
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def plot_losses(self):
        """손실 그래프 시각화"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            save_path = 'training_progress.png'
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 첫 번째 서브플롯: 손실 곡선
            ax1.plot(self.train_losses, label='Training Loss', color='blue', linewidth=2, marker='o', markersize=3)
            ax1.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2, marker='s', markersize=3)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progress (Auto ID별 5분 간격 샘플링)', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # 두 번째 서브플롯: 과적합 모니터
            gaps = [val - train for train, val in zip(self.train_losses, self.val_losses)]
            ax2.plot(gaps, label='Val - Train Gap', color='orange', linewidth=2, marker='^', markersize=3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Gap')
            ax2.set_title('Overfitting Monitor (Gap should be close to 0)', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                print(f"✅ Training progress saved as '{save_path}'")
                print(f"   파일 크기: {os.path.getsize(save_path)} bytes")
            else:
                print("❌ 그래프 저장 실패")
                
            plt.close()
            
        except Exception as e:
            print(f"❌ 그래프 생성 오류: {e}")
            print("손실 데이터를 텍스트로 출력합니다:")
            print(f"훈련 손실: {self.train_losses}")
            print(f"검증 손실: {self.val_losses}")
    
    def create_validation_plots(self, model, test_loader, processor, num_steps=10):
        """스텝별 예측 vs 실제 값 검증 그래프 생성 - 개선된 디버깅 버전"""
        print("🔍 Step-by-step 검증 그래프 생성 시작...")
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os
            
            model.eval()
            
            # 테스트 데이터 수집
            print("📊 테스트 데이터 수집 중...")
            all_states = []
            all_actions = []
            all_targets = []
            
            # 데이터 수집 (더 많은 배치)
            batch_count = 0
            for batch_states, batch_actions, batch_targets in test_loader:
                all_states.append(batch_states)
                all_actions.append(batch_actions) 
                all_targets.append(batch_targets)
                batch_count += 1
                
                if batch_count >= 5:  # 더 많은 배치 수집
                    break
            
            print(f"✅ {batch_count}개 배치 수집 완료")
            
            if len(all_states) < 1:
                print("❌ 테스트 데이터가 부족합니다.")
                return
            
            # 데이터 결합
            states_tensor = torch.cat(all_states, dim=0)
            actions_tensor = torch.cat(all_actions, dim=0) 
            targets_tensor = torch.cat(all_targets, dim=0)
            
            print(f"📊 결합된 데이터 크기:")
            print(f"   States: {states_tensor.shape}")
            print(f"   Actions: {actions_tensor.shape}")
            print(f"   Targets: {targets_tensor.shape}")
            
            # 연속 예측 수행
            print("🔮 연속 예측 수행 중...")
            predictions = []
            actual_values = []
            
            # 더 안전한 시작점 설정
            start_idx = min(10, len(states_tensor) - 1)  # 안전한 시작점
            current_state_seq = states_tensor[start_idx:start_idx+1]
            current_action_seq = actions_tensor[start_idx:start_idx+1]
            
            max_steps = min(num_steps, len(targets_tensor) - start_idx - 1)
            print(f"   시작 인덱스: {start_idx}")
            print(f"   최대 예측 스텝: {max_steps}")
            
            with torch.no_grad():
                for step in range(max_steps):
                    try:
                        # 다음 상태 예측
                        next_state_pred = model(current_state_seq, current_action_seq)
                        predictions.append(next_state_pred[0].numpy())
                        
                        # 실제 다음 상태
                        actual_idx = start_idx + step + 1
                        if actual_idx < len(targets_tensor):
                            actual_values.append(targets_tensor[actual_idx].numpy())
                        
                        # 다음 스텝을 위한 상태 업데이트
                        next_idx = start_idx + step + 1
                        if next_idx < len(states_tensor):
                            current_state_seq = states_tensor[next_idx:next_idx+1]
                            current_action_seq = actions_tensor[next_idx:next_idx+1]
                        
                    except Exception as pred_error:
                        print(f"⚠️ 스텝 {step}에서 예측 오류: {pred_error}")
                        break
            
            print(f"✅ {len(predictions)}개 스텝 예측 완료")
            
            if len(predictions) == 0:
                print("❌ 예측 데이터가 없습니다.")
                return
            
            # 배열 변환
            predictions = np.array(predictions)
            actual_values = np.array(actual_values[:len(predictions)])  # 길이 맞추기
            
            print(f"📊 예측 결과 크기:")
            print(f"   Predictions: {predictions.shape}")
            print(f"   Actual: {actual_values.shape}")
            
            # 역정규화 시도
            try:
                if hasattr(processor, 'state_scaler') and hasattr(processor.state_scaler, 'inverse_transform'):
                    pred_rescaled = processor.state_scaler.inverse_transform(predictions)
                    actual_rescaled = processor.state_scaler.inverse_transform(actual_values)
                    print("✅ 역정규화 완료")
                else:
                    pred_rescaled = predictions.copy()
                    actual_rescaled = actual_values.copy()
                    print("⚠️ 정규화된 값 사용")
            except Exception as e:
                print(f"⚠️ 역정규화 실패: {e}")
                pred_rescaled = predictions.copy()
                actual_rescaled = actual_values.copy()
            
            # 상태 이름 정의 - Power 제거
            state_dim = predictions.shape[1]
            if state_dim == 5:
                state_names = ['Tpip_in_next', 'Tpip_out_next', 'Tbdy_next', 'Tid_next', 'Tod_next']
            elif state_dim == 9:
                state_names = ['Tpip_in1_next', 'Tpip_in2_next', 'Tpip_out1_next', 'Tpip_out2_next', 
                              'Tbdy1_next', 'Tbdy2_next', 'Tid1_next', 'Tid2_next', 'Tod_next']
            else:
                state_names = [f'State_{i+1}_next' for i in range(state_dim)]
            
            print(f"📊 상태 이름: {state_names[:min(5, len(state_names))]}...")
            
            # 그래프 생성
            print("🎨 그래프 생성 중...")
            n_features = min(10, state_dim)
            
            if n_features <= 5:
                rows, cols = 2, 3
            elif n_features <= 8:
                rows, cols = 2, 4
            else:
                rows, cols = 2, 5
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
            fig.suptitle(f'Step-by-Step Prediction vs Actual Validation (Auto ID별 5분 간격)\n({len(predictions)} steps prediction)', 
                        fontsize=16, fontweight='bold')
            
            # axes 처리
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes_flat = axes.flatten()
            
            for i in range(n_features):
                ax = axes_flat[i]
                
                # 스텝별 데이터
                steps = range(len(pred_rescaled))
                true_values = actual_rescaled[:, i]
                pred_values = pred_rescaled[:, i]
                
                # Step plot
                ax.step(steps, true_values, where='post', label='True', color='blue', 
                       linewidth=2, marker='o', markersize=4)
                ax.step(steps, pred_values, where='post', label='Predicted', color='orange', 
                       linewidth=2, marker='s', markersize=4)
                
                ax.set_title(f'{state_names[i]}', fontweight='bold', fontsize=12)
                ax.set_xlabel('Prediction Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # MAE 계산 및 표시
                mae = np.mean(np.abs(pred_values - true_values))
                ax.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=10)
            
            # 빈 서브플롯 숨기기
            for i in range(n_features, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout()
            
            # 저장 시도 (여러 경로)
            print("💾 파일 저장 중...")
            save_paths = [
                'step_by_step_validation.png',
                os.path.join(os.getcwd(), 'step_by_step_validation.png'),
                os.path.join(os.path.expanduser('~'), 'step_by_step_validation.png'),
                os.path.join(os.path.expanduser('~'), 'Desktop', 'step_by_step_validation.png')
            ]
            
            saved = False
            for i, save_path in enumerate(save_paths):
                try:
                    # 디렉토리 확인
                    save_dir = os.path.dirname(save_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    
                    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                    
                    # 파일 존재 및 크기 확인
                    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                        print(f"✅ Step-by-step 검증 그래프 저장: '{save_path}'")
                        print(f"   파일 크기: {os.path.getsize(save_path)} bytes")
                        saved = True
                        break
                    else:
                        print(f"⚠️ 저장 실패 또는 파일 크기 문제: {save_path}")
                        
                except Exception as save_error:
                    print(f"⚠️ 저장 시도 {i+1} 실패: {save_error}")
                    continue
            
            plt.close()
            
            if not saved:
                print("❌ 모든 경로에서 저장 실패")
                print("📊 데이터 요약:")
                overall_mae = np.mean(np.abs(pred_rescaled - actual_rescaled))
                print(f"   전체 MAE: {overall_mae:.3f}")
                print(f"   예측 스텝: {len(predictions)}")
                print(f"   상태 차원: {state_dim}")
            else:
                # 성능 요약
                overall_mae = np.mean(np.abs(pred_rescaled - actual_rescaled))
                print(f"📊 Step-by-step 검증 성능:")
                print(f"   전체 평균 절대 오차: {overall_mae:.3f}")
                print(f"   예측 스텝 수: {len(predictions)}")
                print(f"   상태 차원: {state_dim}")
                
        except Exception as e:
            print(f"❌ Step-by-step 검증 그래프 생성 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def create_single_step_validation(self, model, test_loader, processor, num_samples=5):
        """단일 스텝 예측 vs 실제 값 비교 (Bar Chart 스타일) - 개선된 버전"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            model.eval()
            
            # 테스트 데이터에서 샘플 추출
            sample_states, sample_actions, sample_targets = next(iter(test_loader))
            
            # 예측 수행
            with torch.no_grad():
                predictions = model(sample_states[:num_samples], sample_actions[:num_samples])
            
            predictions_np = predictions.numpy()
            targets_np = sample_targets[:num_samples].numpy()
            
            # 역정규화 시도
            try:
                if hasattr(processor, 'state_scaler'):
                    pred_rescaled = processor.state_scaler.inverse_transform(predictions_np)  
                    actual_rescaled = processor.state_scaler.inverse_transform(targets_np)
                else:
                    pred_rescaled = predictions_np
                    actual_rescaled = targets_np
            except:
                pred_rescaled = predictions_np
                actual_rescaled = targets_np
            
            # 상태 이름 정의 - Power 제거
            state_dim = predictions_np.shape[1]
            if state_dim == 5:
                state_names = ['Tpip_in', 'Tpip_out', 'Tbdy', 'Tid', 'Tod']
            elif state_dim == 9:
                state_names = ['Tpip_in1', 'Tpip_in2', 'Tpip_out1', 'Tpip_out2', 
                              'Tbdy1', 'Tbdy2', 'Tid1', 'Tid2', 'Tod']
            else:
                state_names = [f'State_{i+1}' for i in range(state_dim)]
            
            # 그래프 생성 (3x4 그리드)
            n_features = min(12, state_dim)
            rows = 3
            cols = 4
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
            fig.suptitle('Single-Step Prediction vs Actual Comparison (Auto ID별 5분 간격)\n(각 샘플은 서로 다른 시점의 테스트 데이터)', 
                        fontsize=16, fontweight='bold')
            
            axes_flat = axes.flatten()
            
            for i in range(n_features):
                ax = axes_flat[i]
                
                # 각 샘플에 대해 예측값과 실제값 비교
                pred_values = pred_rescaled[:, i]
                actual_values = actual_rescaled[:, i]
                
                # 바 차트
                x = np.arange(num_samples)
                width = 0.35
                
                bars1 = ax.bar(x - width/2, actual_values, width, label='Actual', 
                              alpha=0.8, color='skyblue')
                bars2 = ax.bar(x + width/2, pred_values, width, label='Predicted', 
                              alpha=0.8, color='lightcoral')
                
                # 오차 표시
                errors = np.abs(pred_values - actual_values)
                for j, (actual, pred, error) in enumerate(zip(actual_values, pred_values, errors)):
                    ax.text(j, max(actual, pred) + abs(max(actual, pred)) * 0.05, 
                           f'{error:.2f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_title(f'{state_names[i]}', fontweight='bold')
                ax.set_xlabel('Test Sample Index\n(각 샘플 = 다른 시점의 데이터)')
                ax.set_ylabel('State Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # x축 라벨을 더 명확하게
                ax.set_xticks(x)
                ax.set_xticklabels([f'Sample\n{i}' for i in range(num_samples)])
            
            # 빈 서브플롯 숨기기
            for i in range(n_features, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout()
            
            # 저장
            save_paths = [
                'single_step_validation.png',
                os.path.join(os.path.expanduser('~'), 'single_step_validation.png'),
                os.path.join(os.path.expanduser('~'), 'Desktop', 'single_step_validation.png')
            ]
            
            for save_path in save_paths:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                    if os.path.exists(save_path):
                        print(f"✅ 단일 스텝 검증 그래프 저장: '{save_path}'")
                        print(f"   📊 X축: 테스트 샘플 인덱스 (0~{num_samples-1})")
                        print(f"   📊 Y축: 각 상태의 값 (예측 vs 실제)")
                        print(f"   📊 막대 위 숫자: 절대 오차")
                        break
                except:
                    continue
            
            plt.close()
            
            # 성능 요약 출력
            print(f"\n📈 단일 스텝 예측 성능:")
            overall_mae = np.mean(np.abs(pred_rescaled - actual_rescaled))
            print(f"   전체 평균 절대 오차: {overall_mae:.3f}")
            print(f"   테스트 샘플 수: {num_samples}")
            print(f"   상태 차원: {state_dim}")
            
        except Exception as e:
            print(f"❌ 단일 스텝 검증 그래프 오류: {e}")
    
    def save_losses_to_csv(self):
        """손실 데이터를 CSV로 저장"""
        try:
            import pandas as pd
            import os
            
            # 데이터프레임 생성
            df = pd.DataFrame({
                'epoch': range(1, len(self.train_losses) + 1),
                'train_loss': self.train_losses,
                'val_loss': self.val_losses,
                'gap': [val - train for train, val in zip(self.train_losses, self.val_losses)]
            })
            
            # 저장 경로들
            save_paths = [
                'training_losses.csv',
                os.path.join(os.path.expanduser('~'), 'training_losses.csv'),
                os.path.join(os.path.expanduser('~'), 'Desktop', 'training_losses.csv')
            ]
            
            for save_path in save_paths:
                try:
                    df.to_csv(save_path, index=False)
                    if os.path.exists(save_path):
                        print(f"✅ 손실 데이터 CSV 저장: '{save_path}'")
                        return save_path
                except:
                    continue
                    
            print("⚠️ CSV 저장 실패")
            return None
        except Exception as e:
            print(f"❌ csv 저장 오류: {e}")
    
    def create_simple_step_validation(self, model, test_loader, processor, num_samples=3):
        """간단한 step-by-step 검증 그래프 (대안 방법)"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os
            
            print("🎨 간단한 step-by-step 검증 그래프 생성...")
            
            model.eval()
            
            # 간단한 방법: 여러 개별 예측을 연결
            sample_states, sample_actions, sample_targets = next(iter(test_loader))
            
            # 첫 몇 개 샘플로 연속 예측 시뮬레이션
            predictions_list = []
            actuals_list = []
            
            with torch.no_grad():
                for i in range(min(num_samples, len(sample_states))):
                    # 각 샘플에 대해 예측
                    pred = model(sample_states[i:i+1], sample_actions[i:i+1])
                    predictions_list.append(pred[0].numpy())
                    actuals_list.append(sample_targets[i].numpy())
            
            if len(predictions_list) == 0:
                print("❌ 예측 데이터가 없습니다.")
                return
            
            predictions = np.array(predictions_list)
            actuals = np.array(actuals_list)
            
            # 역정규화
            try:
                if hasattr(processor, 'state_scaler'):
                    pred_rescaled = processor.state_scaler.inverse_transform(predictions)
                    actual_rescaled = processor.state_scaler.inverse_transform(actuals)
                else:
                    pred_rescaled = predictions
                    actual_rescaled = actuals
            except Exception as e:
                print(f"⚠️ 역정규화 실패: {e}")
                pred_rescaled = predictions
                actual_rescaled = actuals
            
            # 상태 이름 - Power 제거
            state_dim = predictions.shape[1]
            if state_dim == 5:
                state_names = ['Tpip_in', 'Tpip_out', 'Tbdy', 'Tid', 'Tod']
            elif state_dim == 9:
                state_names = ['Tpip_in1', 'Tpip_in2', 'Tpip_out1', 'Tpip_out2', 
                              'Tbdy1', 'Tbdy2', 'Tid1', 'Tid2', 'Tod']
            else:
                state_names = [f'State_{i+1}' for i in range(state_dim)]
            
            # 그래프 생성 (간단한 버전)
            n_features = min(6, state_dim)  # 6개만 표시
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f'Simple Step-by-Step Validation (Auto ID별 5분 간격, {num_samples} steps)', 
                        fontsize=14, fontweight='bold')
            
            axes_flat = axes.flatten()
            
            for i in range(n_features):
                ax = axes_flat[i]
                
                steps = range(len(pred_rescaled))
                true_vals = actual_rescaled[:, i]
                pred_vals = pred_rescaled[:, i]
                
                # 단순한 라인 플롯
                ax.plot(steps, true_vals, 'b-o', label='True', linewidth=2, markersize=6)
                ax.plot(steps, pred_vals, 'r-s', label='Predicted', linewidth=2, markersize=6)
                
                ax.set_title(f'{state_names[i]}', fontweight='bold')
                ax.set_xlabel('Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # MAE
                mae = np.mean(np.abs(pred_vals - true_vals))
                ax.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            plt.tight_layout()
            
            # 저장
            save_path = 'simple_step_validation.png'
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                if os.path.exists(save_path):
                    print(f"✅ 간단한 step-by-step 그래프 저장: '{save_path}'")
                    return save_path
            except Exception as e:
                print(f"❌ 간단한 그래프 저장 실패: {e}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ 간단한 step-by-step 그래프 오류: {e}")
            return None

class ImprovedSSMTrainer(SSMTrainer):
    """개선된 SSM 훈련 클래스 - Lag-1 문제 해결"""
    
    def __init__(self, model: SimpleSSMWorldModel, learning_rate: float = 0.001):
        super().__init__(model, learning_rate)
        # 방향성 정확도를 위한 추가 메트릭
        self.directional_losses = []
        
    def calculate_directional_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                 prev_states: torch.Tensor) -> torch.Tensor:
        """방향성 손실 계산 - 변화 방향을 맞추는지 확인"""
        # 이전 상태 대비 변화량 계산
        pred_changes = predictions - prev_states  # 예측된 변화량
        true_changes = targets - prev_states      # 실제 변화량
        
        # 방향성 일치도 계산 (부호가 같으면 1, 다르면 -1)
        directional_accuracy = torch.sign(pred_changes) * torch.sign(true_changes)
        
        # 방향성 손실 (방향이 틀리면 페널티)
        directional_loss = torch.mean(torch.relu(-directional_accuracy + 0.1))
        
        return directional_loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """한 에포크 훈련 - 개선된 손실 함수"""
        self.model.train()
        total_loss = 0.0
        total_directional_loss = 0.0
        
        for states, actions, targets in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(states, actions)
            
            # 기본 MSE 손실
            mse_loss = self.criterion(predictions, targets)
            
            # 방향성 손실 (이전 상태와 비교)
            prev_states = states[:, -1]  # 마지막 상태를 이전 상태로 사용
            directional_loss = self.calculate_directional_loss(predictions, targets, prev_states)
            
            # 결합된 손실 (MSE + 방향성)
            combined_loss = mse_loss + 0.2 * directional_loss
            
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()
            
            total_loss += mse_loss.item()
            total_directional_loss += directional_loss.item()
            
        avg_loss = total_loss / len(train_loader)
        avg_directional_loss = total_directional_loss / len(train_loader)
        self.directional_losses.append(avg_directional_loss)
        
        return avg_loss
    
    def create_lag1_comparison_plot(self, model, test_loader, processor):
        """Lag-1 예측과 실제 예측 비교 그래프"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os
            
            model.eval()
            
            # 테스트 데이터 가져오기
            sample_states, sample_actions, sample_targets = next(iter(test_loader))
            
            with torch.no_grad():
                predictions = model(sample_states[:10], sample_actions[:10])
                
            predictions_np = predictions.numpy()
            targets_np = sample_targets[:10].numpy()
            prev_states_np = sample_states[:10, -1].numpy()  # 이전 상태
            
            # 역정규화
            try:
                if hasattr(processor, 'state_scaler'):
                    pred_rescaled = processor.state_scaler.inverse_transform(predictions_np)
                    actual_rescaled = processor.state_scaler.inverse_transform(targets_np)
                    prev_rescaled = processor.state_scaler.inverse_transform(prev_states_np)
                else:
                    pred_rescaled = predictions_np
                    actual_rescaled = targets_np
                    prev_rescaled = prev_states_np
            except:
                pred_rescaled = predictions_np
                actual_rescaled = targets_np
                prev_rescaled = prev_states_np
            
            # Lag-1 예측 (이전 값 그대로 사용)
            lag1_predictions = prev_rescaled.copy()
            
            # 첫 번째 상태 (온도) 비교
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Lag-1 Problem Analysis: Real Prediction vs Lag-1 Copy', 
                        fontsize=16, fontweight='bold')
            
            # 실내온도 (Tid) 비교
            temp_idx = 3 if pred_rescaled.shape[1] >= 4 else 0
            
            axes[0, 0].plot(range(10), actual_rescaled[:, temp_idx], 'g-o', 
                           label='Actual', linewidth=2, markersize=6)
            axes[0, 0].plot(range(10), pred_rescaled[:, temp_idx], 'b-s', 
                           label='Model Prediction', linewidth=2, markersize=6)
            axes[0, 0].plot(range(10), lag1_predictions[:, temp_idx], 'r--^', 
                           label='Lag-1 (Previous Value)', linewidth=2, markersize=6)
            axes[0, 0].set_title('Temperature Prediction Comparison')
            axes[0, 0].set_xlabel('Sample Index')
            axes[0, 0].set_ylabel('Temperature (°C)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 변화량 비교
            actual_changes = actual_rescaled[:, temp_idx] - prev_rescaled[:, temp_idx]
            pred_changes = pred_rescaled[:, temp_idx] - prev_rescaled[:, temp_idx]
            lag1_changes = np.zeros_like(actual_changes)  # Lag-1은 변화량이 0
            
            axes[0, 1].bar(np.arange(10) - 0.2, actual_changes, 0.2, 
                          label='Actual Change', alpha=0.7, color='green')
            axes[0, 1].bar(np.arange(10), pred_changes, 0.2, 
                          label='Predicted Change', alpha=0.7, color='blue')
            axes[0, 1].bar(np.arange(10) + 0.2, lag1_changes, 0.2, 
                          label='Lag-1 Change (0)', alpha=0.7, color='red')
            axes[0, 1].set_title('Change Amount Comparison')
            axes[0, 1].set_xlabel('Sample Index')
            axes[0, 1].set_ylabel('Temperature Change (°C)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 오차 비교
            model_errors = np.abs(pred_rescaled[:, temp_idx] - actual_rescaled[:, temp_idx])
            lag1_errors = np.abs(lag1_predictions[:, temp_idx] - actual_rescaled[:, temp_idx])
            
            axes[1, 0].bar(np.arange(10) - 0.2, model_errors, 0.4, 
                          label=f'Model MAE: {np.mean(model_errors):.3f}', 
                          alpha=0.7, color='blue')
            axes[1, 0].bar(np.arange(10) + 0.2, lag1_errors, 0.4, 
                          label=f'Lag-1 MAE: {np.mean(lag1_errors):.3f}', 
                          alpha=0.7, color='red')
            axes[1, 0].set_title('Prediction Error Comparison')
            axes[1, 0].set_xlabel('Sample Index')
            axes[1, 0].set_ylabel('Absolute Error')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 방향성 정확도
            model_directions = np.sign(pred_changes)
            actual_directions = np.sign(actual_changes)
            directional_accuracy = (model_directions == actual_directions).astype(int)
            
            axes[1, 1].bar(range(10), directional_accuracy, alpha=0.7, color='purple')
            axes[1, 1].set_title(f'Directional Accuracy: {np.mean(directional_accuracy)*100:.1f}%')
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Direction Match (1=Correct, 0=Wrong)')
            axes[1, 1].set_ylim([0, 1.2])
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장
            save_path = 'lag1_problem_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            if os.path.exists(save_path):
                print(f"✅ Lag-1 문제 분석 그래프 저장: '{save_path}'")
                print(f"   📊 모델 MAE: {np.mean(model_errors):.3f}")
                print(f"   📊 Lag-1 MAE: {np.mean(lag1_errors):.3f}")  
                print(f"   📊 방향성 정확도: {np.mean(directional_accuracy)*100:.1f}%")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ Lag-1 분석 그래프 생성 오류: {e}")
    """SSM World Model 훈련 클래스"""
    
    def __init__(self, model: SimpleSSMWorldModel, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.7)
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        
        for states, actions, targets in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(states, actions)
            loss = self.criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for states, actions, targets in val_loader:
                predictions = self.model(states, actions)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, patience: int = 10) -> Dict:
        """전체 훈련 프로세스"""
        best_val_loss = float('inf')
        patience_counter = 0
        min_improvement = 1e-4
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            improvement = best_val_loss - val_loss
            
            print(f"Epoch {epoch+1:2d}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                  f"LR: {current_lr:.1e}, Gap: {val_loss-train_loss:.4f}")
            
            if improvement > min_improvement:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_simple_ssm_model.pth')
                print(f"    ✓ New best! Improvement: {improvement:.4f}")
            else:
                patience_counter += 1
                print(f"    No improvement ({patience_counter}/{patience})")
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            if val_loss - train_loss > 0.5:
                print(f"Overfitting detected! Gap: {val_loss - train_loss:.4f}")
                break
                
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def plot_losses(self):
        """손실 그래프 시각화"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            save_path = 'training_progress.png'
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 첫 번째 서브플롯: 손실 곡선
            ax1.plot(self.train_losses, label='Training Loss', color='blue', linewidth=2, marker='o', markersize=3)
            ax1.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2, marker='s', markersize=3)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progress (Auto ID별 5분 간격 샘플링)', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # 두 번째 서브플롯: 과적합 모니터
            gaps = [val - train for train, val in zip(self.train_losses, self.val_losses)]
            ax2.plot(gaps, label='Val - Train Gap', color='orange', linewidth=2, marker='^', markersize=3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Gap')
            ax2.set_title('Overfitting Monitor (Gap should be close to 0)', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                print(f"✅ Training progress saved as '{save_path}'")
                print(f"   파일 크기: {os.path.getsize(save_path)} bytes")
            else:
                print("❌ 그래프 저장 실패")
                
            plt.close()
            
        except Exception as e:
            print(f"❌ 그래프 생성 오류: {e}")
            print("손실 데이터를 텍스트로 출력합니다:")
            print(f"훈련 손실: {self.train_losses}")
            print(f"검증 손실: {self.val_losses}")
    
    def create_validation_plots(self, model, test_loader, processor, num_steps=10):
        """스텝별 예측 vs 실제 값 검증 그래프 생성 - 개선된 디버깅 버전"""
        print("🔍 Step-by-step 검증 그래프 생성 시작...")
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os
            
            model.eval()
            
            # 테스트 데이터 수집
            print("📊 테스트 데이터 수집 중...")
            all_states = []
            all_actions = []
            all_targets = []
            
            # 데이터 수집 (더 많은 배치)
            batch_count = 0
            for batch_states, batch_actions, batch_targets in test_loader:
                all_states.append(batch_states)
                all_actions.append(batch_actions) 
                all_targets.append(batch_targets)
                batch_count += 1
                
                if batch_count >= 5:  # 더 많은 배치 수집
                    break
            
            print(f"✅ {batch_count}개 배치 수집 완료")
            
            if len(all_states) < 1:
                print("❌ 테스트 데이터가 부족합니다.")
                return
            
            # 데이터 결합
            states_tensor = torch.cat(all_states, dim=0)
            actions_tensor = torch.cat(all_actions, dim=0) 
            targets_tensor = torch.cat(all_targets, dim=0)
            
            print(f"📊 결합된 데이터 크기:")
            print(f"   States: {states_tensor.shape}")
            print(f"   Actions: {actions_tensor.shape}")
            print(f"   Targets: {targets_tensor.shape}")
            
            # 연속 예측 수행
            print("🔮 연속 예측 수행 중...")
            predictions = []
            actual_values = []
            
            # 더 안전한 시작점 설정
            start_idx = min(10, len(states_tensor) - 1)  # 안전한 시작점
            current_state_seq = states_tensor[start_idx:start_idx+1]
            current_action_seq = actions_tensor[start_idx:start_idx+1]
            
            max_steps = min(num_steps, len(targets_tensor) - start_idx - 1)
            print(f"   시작 인덱스: {start_idx}")
            print(f"   최대 예측 스텝: {max_steps}")
            
            with torch.no_grad():
                for step in range(max_steps):
                    try:
                        # 다음 상태 예측
                        next_state_pred = model(current_state_seq, current_action_seq)
                        predictions.append(next_state_pred[0].numpy())
                        
                        # 실제 다음 상태
                        actual_idx = start_idx + step + 1
                        if actual_idx < len(targets_tensor):
                            actual_values.append(targets_tensor[actual_idx].numpy())
                        
                        # 다음 스텝을 위한 상태 업데이트
                        next_idx = start_idx + step + 1
                        if next_idx < len(states_tensor):
                            current_state_seq = states_tensor[next_idx:next_idx+1]
                            current_action_seq = actions_tensor[next_idx:next_idx+1]
                        
                    except Exception as pred_error:
                        print(f"⚠️ 스텝 {step}에서 예측 오류: {pred_error}")
                        break
            
            print(f"✅ {len(predictions)}개 스텝 예측 완료")
            
            if len(predictions) == 0:
                print("❌ 예측 데이터가 없습니다.")
                return
            
            # 배열 변환
            predictions = np.array(predictions)
            actual_values = np.array(actual_values[:len(predictions)])  # 길이 맞추기
            
            print(f"📊 예측 결과 크기:")
            print(f"   Predictions: {predictions.shape}")
            print(f"   Actual: {actual_values.shape}")
            
            # 역정규화 시도
            try:
                if hasattr(processor, 'state_scaler') and hasattr(processor.state_scaler, 'inverse_transform'):
                    pred_rescaled = processor.state_scaler.inverse_transform(predictions)
                    actual_rescaled = processor.state_scaler.inverse_transform(actual_values)
                    print("✅ 역정규화 완료")
                else:
                    pred_rescaled = predictions.copy()
                    actual_rescaled = actual_values.copy()
                    print("⚠️ 정규화된 값 사용")
            except Exception as e:
                print(f"⚠️ 역정규화 실패: {e}")
                pred_rescaled = predictions.copy()
                actual_rescaled = actual_values.copy()
            
            # 상태 이름 정의 - Power 제거
            state_dim = predictions.shape[1]
            if state_dim == 5:
                state_names = ['Tpip_in_next', 'Tpip_out_next', 'Tbdy_next', 'Tid_next', 'Tod_next']
            elif state_dim == 9:
                state_names = ['Tpip_in1_next', 'Tpip_in2_next', 'Tpip_out1_next', 'Tpip_out2_next', 
                              'Tbdy1_next', 'Tbdy2_next', 'Tid1_next', 'Tid2_next', 'Tod_next']
            else:
                state_names = [f'State_{i+1}_next' for i in range(state_dim)]
            
            print(f"📊 상태 이름: {state_names[:min(5, len(state_names))]}...")
            
            # 그래프 생성
            print("🎨 그래프 생성 중...")
            n_features = min(10, state_dim)
            
            if n_features <= 5:
                rows, cols = 2, 3
            elif n_features <= 8:
                rows, cols = 2, 4
            else:
                rows, cols = 2, 5
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
            fig.suptitle(f'Step-by-Step Prediction vs Actual Validation (Auto ID별 5분 간격)\n({len(predictions)} steps prediction)', 
                        fontsize=16, fontweight='bold')
            
            # axes 처리
            if rows == 1:
                axes = axes.reshape(1, -1)
            axes_flat = axes.flatten()
            
            for i in range(n_features):
                ax = axes_flat[i]
                
                # 스텝별 데이터
                steps = range(len(pred_rescaled))
                true_values = actual_rescaled[:, i]
                pred_values = pred_rescaled[:, i]
                
                # Step plot
                ax.step(steps, true_values, where='post', label='True', color='blue', 
                       linewidth=2, marker='o', markersize=4)
                ax.step(steps, pred_values, where='post', label='Predicted', color='orange', 
                       linewidth=2, marker='s', markersize=4)
                
                ax.set_title(f'{state_names[i]}', fontweight='bold', fontsize=12)
                ax.set_xlabel('Prediction Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # MAE 계산 및 표시
                mae = np.mean(np.abs(pred_values - true_values))
                ax.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=10)
            
            # 빈 서브플롯 숨기기
            for i in range(n_features, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout()
            
            # 저장 시도 (여러 경로)
            print("💾 파일 저장 중...")
            save_paths = [
                'step_by_step_validation.png',
                os.path.join(os.getcwd(), 'step_by_step_validation.png'),
                os.path.join(os.path.expanduser('~'), 'step_by_step_validation.png'),
                os.path.join(os.path.expanduser('~'), 'Desktop', 'step_by_step_validation.png')
            ]
            
            saved = False
            for i, save_path in enumerate(save_paths):
                try:
                    # 디렉토리 확인
                    save_dir = os.path.dirname(save_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    
                    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                    
                    # 파일 존재 및 크기 확인
                    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
                        print(f"✅ Step-by-step 검증 그래프 저장: '{save_path}'")
                        print(f"   파일 크기: {os.path.getsize(save_path)} bytes")
                        saved = True
                        break
                    else:
                        print(f"⚠️ 저장 실패 또는 파일 크기 문제: {save_path}")
                        
                except Exception as save_error:
                    print(f"⚠️ 저장 시도 {i+1} 실패: {save_error}")
                    continue
            
            plt.close()
            
            if not saved:
                print("❌ 모든 경로에서 저장 실패")
                print("📊 데이터 요약:")
                overall_mae = np.mean(np.abs(pred_rescaled - actual_rescaled))
                print(f"   전체 MAE: {overall_mae:.3f}")
                print(f"   예측 스텝: {len(predictions)}")
                print(f"   상태 차원: {state_dim}")
            else:
                # 성능 요약
                overall_mae = np.mean(np.abs(pred_rescaled - actual_rescaled))
                print(f"📊 Step-by-step 검증 성능:")
                print(f"   전체 평균 절대 오차: {overall_mae:.3f}")
                print(f"   예측 스텝 수: {len(predictions)}")
                print(f"   상태 차원: {state_dim}")
                
        except Exception as e:
            print(f"❌ Step-by-step 검증 그래프 생성 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def create_single_step_validation(self, model, test_loader, processor, num_samples=5):
        """단일 스텝 예측 vs 실제 값 비교 (Bar Chart 스타일) - 개선된 버전"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            model.eval()
            
            # 테스트 데이터에서 샘플 추출
            sample_states, sample_actions, sample_targets = next(iter(test_loader))
            
            # 예측 수행
            with torch.no_grad():
                predictions = model(sample_states[:num_samples], sample_actions[:num_samples])
            
            predictions_np = predictions.numpy()
            targets_np = sample_targets[:num_samples].numpy()
            
            # 역정규화 시도
            try:
                if hasattr(processor, 'state_scaler'):
                    pred_rescaled = processor.state_scaler.inverse_transform(predictions_np)  
                    actual_rescaled = processor.state_scaler.inverse_transform(targets_np)
                else:
                    pred_rescaled = predictions_np
                    actual_rescaled = targets_np
            except:
                pred_rescaled = predictions_np
                actual_rescaled = targets_np
            
            # 상태 이름 정의 - Power 제거
            state_dim = predictions_np.shape[1]
            if state_dim == 5:
                state_names = ['Tpip_in', 'Tpip_out', 'Tbdy', 'Tid', 'Tod']
            elif state_dim == 9:
                state_names = ['Tpip_in1', 'Tpip_in2', 'Tpip_out1', 'Tpip_out2', 
                              'Tbdy1', 'Tbdy2', 'Tid1', 'Tid2', 'Tod']
            else:
                state_names = [f'State_{i+1}' for i in range(state_dim)]
            
            # 그래프 생성 (3x4 그리드)
            n_features = min(12, state_dim)
            rows = 3
            cols = 4
            
            fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
            fig.suptitle('Single-Step Prediction vs Actual Comparison (Auto ID별 5분 간격)\n(각 샘플은 서로 다른 시점의 테스트 데이터)', 
                        fontsize=16, fontweight='bold')
            
            axes_flat = axes.flatten()
            
            for i in range(n_features):
                ax = axes_flat[i]
                
                # 각 샘플에 대해 예측값과 실제값 비교
                pred_values = pred_rescaled[:, i]
                actual_values = actual_rescaled[:, i]
                
                # 바 차트
                x = np.arange(num_samples)
                width = 0.35
                
                bars1 = ax.bar(x - width/2, actual_values, width, label='Actual', 
                              alpha=0.8, color='skyblue')
                bars2 = ax.bar(x + width/2, pred_values, width, label='Predicted', 
                              alpha=0.8, color='lightcoral')
                
                # 오차 표시
                errors = np.abs(pred_values - actual_values)
                for j, (actual, pred, error) in enumerate(zip(actual_values, pred_values, errors)):
                    ax.text(j, max(actual, pred) + abs(max(actual, pred)) * 0.05, 
                           f'{error:.2f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_title(f'{state_names[i]}', fontweight='bold')
                ax.set_xlabel('Test Sample Index\n(각 샘플 = 다른 시점의 데이터)')
                ax.set_ylabel('State Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # x축 라벨을 더 명확하게
                ax.set_xticks(x)
                ax.set_xticklabels([f'Sample\n{i}' for i in range(num_samples)])
            
            # 빈 서브플롯 숨기기
            for i in range(n_features, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout()
            
            # 저장
            save_paths = [
                'single_step_validation.png',
                os.path.join(os.path.expanduser('~'), 'single_step_validation.png'),
                os.path.join(os.path.expanduser('~'), 'Desktop', 'single_step_validation.png')
            ]
            
            for save_path in save_paths:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                    if os.path.exists(save_path):
                        print(f"✅ 단일 스텝 검증 그래프 저장: '{save_path}'")
                        print(f"   📊 X축: 테스트 샘플 인덱스 (0~{num_samples-1})")
                        print(f"   📊 Y축: 각 상태의 값 (예측 vs 실제)")
                        print(f"   📊 막대 위 숫자: 절대 오차")
                        break
                except:
                    continue
            
            plt.close()
            
            # 성능 요약 출력
            print(f"\n📈 단일 스텝 예측 성능:")
            overall_mae = np.mean(np.abs(pred_rescaled - actual_rescaled))
            print(f"   전체 평균 절대 오차: {overall_mae:.3f}")
            print(f"   테스트 샘플 수: {num_samples}")
            print(f"   상태 차원: {state_dim}")
            
        except Exception as e:
            print(f"❌ 단일 스텝 검증 그래프 오류: {e}")
    
    def save_losses_to_csv(self):
        """손실 데이터를 CSV로 저장"""
        try:
            import pandas as pd
            import os
            
            # 데이터프레임 생성
            df = pd.DataFrame({
                'epoch': range(1, len(self.train_losses) + 1),
                'train_loss': self.train_losses,
                'val_loss': self.val_losses,
                'gap': [val - train for train, val in zip(self.train_losses, self.val_losses)]
            })
            
            # 저장 경로들
            save_paths = [
                'training_losses.csv',
                os.path.join(os.path.expanduser('~'), 'training_losses.csv'),
                os.path.join(os.path.expanduser('~'), 'Desktop', 'training_losses.csv')
            ]
            
            for save_path in save_paths:
                try:
                    df.to_csv(save_path, index=False)
                    if os.path.exists(save_path):
                        print(f"✅ 손실 데이터 CSV 저장: '{save_path}'")
                        return save_path
                except:
                    continue
                    
            print("⚠️ CSV 저장 실패")
            return None
        except Exception as e:
            print(f"❌ csv 저장 오류: {e}")
    
    def create_simple_step_validation(self, model, test_loader, processor, num_samples=3):
        """간단한 step-by-step 검증 그래프 (대안 방법)"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import os
            
            print("🎨 간단한 step-by-step 검증 그래프 생성...")
            
            model.eval()
            
            # 간단한 방법: 여러 개별 예측을 연결
            sample_states, sample_actions, sample_targets = next(iter(test_loader))
            
            # 첫 몇 개 샘플로 연속 예측 시뮬레이션
            predictions_list = []
            actuals_list = []
            
            with torch.no_grad():
                for i in range(min(num_samples, len(sample_states))):
                    # 각 샘플에 대해 예측
                    pred = model(sample_states[i:i+1], sample_actions[i:i+1])
                    predictions_list.append(pred[0].numpy())
                    actuals_list.append(sample_targets[i].numpy())
            
            if len(predictions_list) == 0:
                print("❌ 예측 데이터가 없습니다.")
                return
            
            predictions = np.array(predictions_list)
            actuals = np.array(actuals_list)
            
            # 역정규화
            try:
                if hasattr(processor, 'state_scaler'):
                    pred_rescaled = processor.state_scaler.inverse_transform(predictions)
                    actual_rescaled = processor.state_scaler.inverse_transform(actuals)
                else:
                    pred_rescaled = predictions
                    actual_rescaled = actuals
            except Exception as e:
                print(f"⚠️ 역정규화 실패: {e}")
                pred_rescaled = predictions
                actual_rescaled = actuals
            
            # 상태 이름 - Power 제거
            state_dim = predictions.shape[1]
            if state_dim == 5:
                state_names = ['Tpip_in', 'Tpip_out', 'Tbdy', 'Tid', 'Tod']
            elif state_dim == 9:
                state_names = ['Tpip_in1', 'Tpip_in2', 'Tpip_out1', 'Tpip_out2', 
                              'Tbdy1', 'Tbdy2', 'Tid1', 'Tid2', 'Tod']
            else:
                state_names = [f'State_{i+1}' for i in range(state_dim)]
            
            # 그래프 생성 (간단한 버전)
            n_features = min(6, state_dim)  # 6개만 표시
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f'Simple Step-by-Step Validation (Auto ID별 5분 간격, {num_samples} steps)', 
                        fontsize=14, fontweight='bold')
            
            axes_flat = axes.flatten()
            
            for i in range(n_features):
                ax = axes_flat[i]
                
                steps = range(len(pred_rescaled))
                true_vals = actual_rescaled[:, i]
                pred_vals = pred_rescaled[:, i]
                
                # 단순한 라인 플롯
                ax.plot(steps, true_vals, 'b-o', label='True', linewidth=2, markersize=6)
                ax.plot(steps, pred_vals, 'r-s', label='Predicted', linewidth=2, markersize=6)
                
                ax.set_title(f'{state_names[i]}', fontweight='bold')
                ax.set_xlabel('Step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # MAE
                mae = np.mean(np.abs(pred_vals - true_vals))
                ax.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            plt.tight_layout()
            
            # 저장
            save_path = 'simple_step_validation.png'
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                if os.path.exists(save_path):
                    print(f"✅ 간단한 step-by-step 그래프 저장: '{save_path}'")
                    return save_path
            except Exception as e:
                print(f"❌ 간단한 그래프 저장 실패: {e}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ 간단한 step-by-step 그래프 오류: {e}")
            return None

def main():
    """메인 실행 함수 - Auto ID별 5분 간격 샘플링 수정 버전"""
    
    # matplotlib 설정
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()
        print("✅ matplotlib 백엔드 설정 완료")
    except Exception as e:
        print(f"⚠️ matplotlib 설정 문제: {e}")
    
    print("🔥 HVAC SSM World Model 훈련 시작! (Auto ID별 5분 간격 샘플링)")
    
    # 1. 데이터 로드 및 전처리
    print("\n" + "="*50)
    print("📊 데이터 로드 및 전처리")
    print("="*50)
    
    processor = HVACDataProcessor()
    file_pattern = "../경남_사무실/LOG_SMARTCARE_*.csv"
    
    try:
        df = processor.load_data(file_pattern)
        print(f"✅ 데이터 로드 완료: {df.shape}")
        
        # Auto ID별 데이터 분포 확인
        auto_id_counts = df['Auto Id'].value_counts().sort_index()
        print(f"📊 Auto ID별 데이터 분포 (5분 간격 리샘플링 후):")
        for auto_id, count in auto_id_counts.items():
            print(f"   Auto ID {auto_id}: {count}개")
        
        # 특성 추출
        states, actions, file_sources = processor.extract_features(df)
        print(f"✅ 특성 추출 완료")
        print(f"   States: {states.shape}")
        print(f"   Actions: {actions.shape}")
        
        # 시퀀스 생성 (Auto ID별 독립적)
        seq_length = 10
        X_states, X_actions, y_states = processor.create_sequences_per_auto_id(
            states, actions, file_sources, seq_length
        )
        print(f"✅ 시퀀스 생성 완료 (Auto ID별 독립적)")
        print(f"   X_states: {X_states.shape}")
        print(f"   X_actions: {X_actions.shape}")
        print(f"   y_states: {y_states.shape}")
        
    except Exception as e:
        print(f"⚠️ 실제 데이터 로드 실패: {e}")
        print("더미 데이터로 진행합니다...")
        
        # 더미 데이터 생성 (Auto ID별로 구분) - Power 제거
        n_auto_ids = 6  # Auto ID 0~5
        n_samples_per_id = 1000
        seq_length = 10
        
        np.random.seed(42)
        
        all_states = []
        all_actions = []
        all_file_sources = []
        
        for auto_id in range(n_auto_ids):
            # Auto ID별로 약간 다른 특성을 가진 더미 데이터
            base_temp = 25 + np.random.randn(n_samples_per_id) * 3 + auto_id * 0.5
            
            states = np.column_stack([
                base_temp + np.random.randn(n_samples_per_id) * 1,  # Tpip_in
                base_temp + np.random.randn(n_samples_per_id) * 1,  # Tpip_out  
                base_temp + np.random.randn(n_samples_per_id) * 0.5,  # Tbdy
                base_temp + np.random.randn(n_samples_per_id) * 2,  # Tid
                base_temp + 5 + np.sin(np.arange(n_samples_per_id) * 0.01) * 3  # Tod (Power 제거)
            ])
            
            tcon = 22 + np.random.randn(n_samples_per_id) * 2
            on_off = (np.abs(tcon - states[:, 3]) > 1).astype(int)
            ptarget = 20 + 5.0 + np.random.randn(n_samples_per_id) * 1  # Power 없으므로 고정값 사용
            
            actions = np.column_stack([tcon, on_off, ptarget])
            
            file_sources = np.repeat(f'dummy_file_AutoID_{auto_id}', n_samples_per_id)
            
            all_states.append(states)
            all_actions.append(actions)
            all_file_sources.append(file_sources)
            
            print(f"   Auto ID {auto_id}: State 차원 {states.shape[1]}개 (Power 제외)")
        
        # 모든 Auto ID 데이터 결합
        states = np.vstack(all_states)
        actions = np.vstack(all_actions)
        file_sources = np.concatenate(all_file_sources)
        
        processor = HVACDataProcessor()
        X_states, X_actions, y_states = processor.create_sequences_per_auto_id(
            states, actions, file_sources, seq_length
        )
        
        print(f"✅ 더미 데이터 생성 완료 (Auto ID별 구분)")
        print(f"   Auto ID 개수: {n_auto_ids}")
        print(f"   각 Auto ID당 샘플: {n_samples_per_id}")
        print(f"   X_states: {X_states.shape}")
        print(f"   X_actions: {X_actions.shape}")
        print(f"   y_states: {y_states.shape}")
    
    # 2. 데이터셋 분할
    print("\n" + "="*50)
    print("🔄 데이터셋 분할")
    print("="*50)
    
    X_states_train, X_states_test, X_actions_train, X_actions_test, y_states_train, y_states_test = train_test_split(
        X_states, X_actions, y_states, test_size=0.2, random_state=42
    )
    
    X_states_train, X_states_val, X_actions_train, X_actions_val, y_states_train, y_states_val = train_test_split(
        X_states_train, X_actions_train, y_states_train, test_size=0.25, random_state=42
    )
    
    print(f"✅ 데이터셋 분할 완료")
    print(f"   훈련: {len(X_states_train)}개")
    print(f"   검증: {len(X_states_val)}개") 
    print(f"   테스트: {len(X_states_test)}개")
    
    # 3. 데이터 로더 생성
    batch_size = 32
    
    train_dataset = HVACDataset(X_states_train, X_actions_train, y_states_train)
    val_dataset = HVACDataset(X_states_val, X_actions_val, y_states_val)
    test_dataset = HVACDataset(X_states_test, X_actions_test, y_states_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 4. 모델 생성
    print("\n" + "="*50)
    print("🤖 모델 생성")
    print("="*50)
    
    state_dim = X_states.shape[-1]
    action_dim = X_actions.shape[-1]
    
    model = SimpleSSMWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=32,
        latent_dim=16
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 모델 생성 완료")
    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   총 파라미터: {total_params:,}개")
    print(f"   모델 크기: ~{total_params * 4 / 1024:.1f}KB")
    
    # 5. 모델 훈련
    print("\n" + "="*50)
    print("🎯 모델 훈련")
    print("="*50)
    
    trainer = SSMTrainer(model, learning_rate=0.003)
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=200,
        patience=10
    )
    
    # 6. 결과 시각화
    print("\n" + "="*50)
    print("📈 결과 시각화")
    print("="*50)
    
    # 그래프 저장 시도
    trainer.plot_losses()
    
    # CSV로도 저장 (안전한 호출)
    try:
        csv_path = trainer.save_losses_to_csv()
        if csv_path:
            print(f"📊 CSV 데이터도 저장됨: {csv_path}")
    except Exception as e:
        print(f"⚠️ CSV 저장 중 오류: {e}")
        csv_path = None
    
    # 시각화 문제 해결을 위한 추가 방법들
    print("\n🔧 시각화 문제 해결 방법:")
    print("1. 현재 디렉토리에서 'training_progress.png' 파일 확인")
    print("2. 바탕화면에서 'training_progress.png' 파일 확인") 
    print("3. 시스템 임시 폴더에서 파일 확인")
    if csv_path:
        print(f"4. CSV 데이터 파일: {csv_path}")
    
    # 손실 데이터 콘솔 출력
    print(f"\n📊 훈련 진행 상황:")
    print(f"   총 에포크: {len(results['train_losses'])}")
    print(f"   최종 훈련 손실: {results['train_losses'][-1]:.6f}")
    print(f"   최종 검증 손실: {results['val_losses'][-1]:.6f}")
    print(f"   최종 Gap: {results['val_losses'][-1] - results['train_losses'][-1]:.6f}")
    
    # 간단한 ASCII 그래프 생성
    print(f"\n📈 손실 추이 (ASCII):")
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    
    # 최근 10개 에포크만 표시
    recent_epochs = min(10, len(train_losses))
    print(f"   최근 {recent_epochs}개 에포크:")
    
    for i in range(len(train_losses) - recent_epochs, len(train_losses)):
        epoch = i + 1
        train_loss = train_losses[i]
        val_loss = val_losses[i]
        
        # 간단한 바 표시 (0-1 스케일)
        max_loss = max(max(train_losses), max(val_losses))
        train_bar = int((train_loss / max_loss) * 20)
        val_bar = int((val_loss / max_loss) * 20)
        
        print(f"   Epoch {epoch:2d}: Train {'█' * train_bar:<20} {train_loss:.4f}")
        print(f"            Val   {'█' * val_bar:<20} {val_loss:.4f}")
        print("")
    
    # 7. 테스트 성능 평가 및 검증 그래프
    print("\n" + "="*50)
    print("🔍 모델 검증 및 성능 평가")
    print("="*50)
    
    test_loss = trainer.validate(test_loader)
    print(f"✅ 최종 테스트 손실: {test_loss:.6f}")
    
    # 검증 그래프 생성
    print("\n📊 검증 그래프 생성 중...")
    
    try:
        # 1. Step-by-step 예측 검증 (이미지와 같은 스타일)
        trainer.create_validation_plots(model, test_loader, processor, num_steps=10)
        print("✅ Step-by-step 검증 그래프 생성 완료")
    except Exception as e:
        print(f"⚠️ Step-by-step 검증 그래프 생성 실패: {e}")
        
        # 대안: 간단한 step-by-step 그래프 생성
        try:
            print("🔄 간단한 step-by-step 그래프 생성 시도...")
            trainer.create_simple_step_validation(model, test_loader, processor)
        except Exception as e2:
            print(f"⚠️ 간단한 검증 그래프도 실패: {e2}")
    
    try:
        # 2. 단일 스텝 예측 vs 실제 (Bar chart 스타일)  
        trainer.create_single_step_validation(model, test_loader, processor, num_samples=5)
        print("✅ 단일 스텝 검증 그래프 생성 완료")
    except Exception as e:
        print(f"⚠️ 단일 스텝 검증 그래프 생성 실패: {e}")
    
    print("📁 저장된 파일들:")
    print("   - training_progress.png: 훈련 과정 그래프")
    print("   - step_by_step_validation.png: 연속 예측 검증")
    print("   - single_step_validation.png: 단일 스텝 예측 검증")
    if csv_path:
        print(f"   - {os.path.basename(csv_path)}: 손실 데이터 CSV")
    
    # 추가: 수동 그래프 생성 함수
    def create_manual_plot():
        """수동으로 그래프 생성 및 저장"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # 현재 작업 디렉토리에 저장
            current_dir = os.getcwd()
            save_path = os.path.join(current_dir, 'hvac_training_results.png')
            
            print(f"🎨 수동 그래프 생성 중... 저장 경로: {save_path}")
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # 훈련/검증 손실
            axes[0].plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Training Loss', markersize=4)
            axes[0].plot(range(1, len(val_losses)+1), val_losses, 'r-s', label='Validation Loss', markersize=4)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('HVAC SSM Training Progress (Auto ID별 5분 간격)', fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_yscale('log')
            
            # Gap 분석
            gaps = [v - t for t, v in zip(train_losses, val_losses)]
            axes[1].plot(range(1, len(gaps)+1), gaps, 'orange', linewidth=2, marker='^', markersize=4)
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Validation - Training Loss')
            axes[1].set_title('Overfitting Monitor', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 여러 형식으로 저장 시도
            formats = ['png', 'pdf', 'svg']
            for fmt in formats:
                try:
                    file_path = save_path.replace('.png', f'.{fmt}')
                    plt.savefig(file_path, format=fmt, dpi=300, bbox_inches='tight')
                    if os.path.exists(file_path):
                        print(f"✅ 그래프 저장 성공: {file_path}")
                        break
                except Exception as e:
                    print(f"⚠️ {fmt} 저장 실패: {e}")
            
            plt.close()
            
        except Exception as e:
            print(f"❌ 수동 그래프 생성 실패: {e}")
    
    # 수동 그래프 생성 시도
    if len(results['train_losses']) > 0:
        create_manual_plot()
    
    # 8. 예측 예시 (콘솔 출력)
    print("\n" + "="*50)
    print("🔮 예측 예시")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        sample_states, sample_actions, sample_targets = next(iter(test_loader))
        predictions = model(sample_states[:3], sample_actions[:3])
        
        print("예측 vs 실제 (정규화된 값, 첫 3개 샘플):")
        for i in range(3):
            pred = predictions[i].numpy()
            target = sample_targets[i].numpy()
            error = np.abs(pred - target)
            
            print(f"\n샘플 {i+1}:")
            print(f"  예측값: [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}...] (총 {len(pred)}개)")
            print(f"  실제값: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}...] (총 {len(target)}개)")
            print(f"  최대오차: {np.max(error):.4f}")
    
    # 9. 강화학습 통합 가이드
    print("\n" + "="*60)
    print("🎮 강화학습 통합 가이드 (Auto ID별 5분 간격 데이터 기반)")
    print("="*60)
    
    class HVACEnvironmentModel:
        def __init__(self, world_model, state_scaler, action_scaler):
            self.world_model = world_model
            self.state_scaler = state_scaler
            self.action_scaler = action_scaler
            
        def predict_next_state(self, current_state, action):
            # 정규화
            state_norm = self.state_scaler.transform(current_state.reshape(1, -1))[0]
            action_norm = self.action_scaler.transform(action.reshape(1, -1))[0]
            
            # 예측
            next_state_norm = self.world_model.predict_next_state(
                torch.FloatTensor(state_norm), 
                torch.FloatTensor(action_norm)
            ).numpy()
            
            # 역정규화
            next_state = self.state_scaler.inverse_transform(next_state_norm.reshape(1, -1))[0]
            return next_state
            
        def calculate_reward(self, state, action, next_state):
            # 개선된 보상 함수 (5분 간격 데이터 고려, Power 제외)
            if len(state) >= 4:
                temp_error = abs(next_state[3] - action[0])  # 목표온도와의 차이 (Tid vs Tcon)
                
                # 5분 간격이므로 변화율 고려
                temp_change_rate = abs(next_state[3] - state[3]) / 5.0  # 분당 온도 변화율
                stability_bonus = -temp_change_rate if temp_change_rate > 1.0 else 0.1
                
                # 배관 온도 효율성 (Tpip_in, Tpip_out 관계)
                pipe_efficiency = -abs(next_state[0] - next_state[1])  # 배관 온도차가 적을수록 효율적
                
                reward = -temp_error + stability_bonus + pipe_efficiency * 0.1
            else:
                reward = -np.mean(np.abs(next_state - state))
            
            return reward
    
    env_model = HVACEnvironmentModel(model, processor.state_scaler, processor.action_scaler)
    
    print("✅ 강화학습 환경 모델 생성 완료 (5분 간격 데이터 기반)")
    print("\n🎯 주요 개선사항:")
    print("- Auto ID별로 독립적인 5분 간격 데이터 사용")
    print("- Power 제외: 측정 주기가 달라 World Model에서 제외")
    print("- 온도 중심의 보상 함수: 배관 효율성 및 안정성 고려")
    print("- 실제 에어컨 운영 패턴 반영")
    
    print(f"\n📊 최종 성능 요약:")
    print(f"   최종 검증 손실: {results['best_val_loss']:.6f}")
    print(f"   최종 테스트 손실: {test_loss:.6f}")
    print(f"   모델 파라미터: {total_params:,}개")
    
    return {
        'best_val_loss': results['best_val_loss'],
        'test_loss': test_loss,
        'model_size_kb': total_params * 4 / 1024,
        'total_params': total_params
    }

if __name__ == "__main__":
    try:
        performance_metrics = main()
        print(f"\n🎉 프로그램 완료! (Auto ID별 5분 간격 샘플링)")
        print(f"   Test Loss: {performance_metrics['test_loss']:.6f}")
        print(f"   Model Size: {performance_metrics['model_size_kb']:.1f}KB")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()