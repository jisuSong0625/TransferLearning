import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class AirConditionerDataProcessor:
    """에어컨 로그 데이터를 World Models 형태로 전처리"""
    
    def __init__(self, data_path="../경남_사무실/LOG_SMARTCARE_*.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_and_process_data(self):
        """CSV 파일들을 로드하고 전처리"""
        files = glob.glob(self.data_path)
        all_data = []
        
        for file in files:
            df = pd.read_csv(file)
            processed_df = self._process_single_file(df)
            all_data.append(processed_df)
            
        combined_data = pd.concat(all_data, ignore_index=True)
        return self._create_state_action_pairs(combined_data)
    
    def _process_single_file(self, df):
        """단일 파일 전처리"""
        # 시간 컬럼을 datetime으로 변환
        df['Time'] = pd.to_datetime(df['Time'])
        
        # 에어컨 그룹 생성: (Auto Id+1)//2
        df['AC_Group'] = (df['Auto Id'] + 1) // 2
        
        # 에어컨 번호 생성: 1 if (Auto Id%2)==1, 2 if (Auto Id%2)==0
        df['AC_Number'] = df['Auto Id'].apply(lambda x: 1 if x % 2 == 1 else 2)
        
        return df
    
    def _create_state_action_pairs(self, df):
        """State와 Action 변수 생성"""
        processed_data = []
        
        # 각 AC 그룹별로 처리
        for group in df['AC_Group'].unique():
            group_data = df[df['AC_Group'] == group].copy()
            
            # AC1, AC2 데이터 분리
            ac1_data = group_data[group_data['AC_Number'] == 1]
            ac2_data = group_data[group_data['AC_Number'] == 2]
            
            if len(ac1_data) == 0 or len(ac2_data) == 0:
                continue
                
            # 시간 순 정렬
            ac1_data = ac1_data.sort_values('Time')
            ac2_data = ac2_data.sort_values('Time')
            
            # 시간 매칭 (가장 가까운 시간끼리 매칭)
            matched_data = self._match_time_series(ac1_data, ac2_data, group)
            processed_data.append(matched_data)
        
        if processed_data:
            return pd.concat(processed_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _match_time_series(self, ac1_data, ac2_data, group):
        """두 에어컨의 시계열 데이터 매칭"""
        matched_rows = []
        
        for _, row1 in ac1_data.iterrows():
            # 가장 가까운 시간의 AC2 데이터 찾기
            time_diff = abs(ac2_data['Time'] - row1['Time'])
            closest_idx = time_diff.idxmin()
            row2 = ac2_data.loc[closest_idx]
            
            # State 변수들 (현재 상태)
            state = {
                'Time': row1['Time'],
                'AC_Group': group,
                'Tpip_in1': row1.get('Tpip_in', np.nan),
                'Tpip_in2': row2.get('Tpip_in', np.nan),
                'Tpip_out1': row1.get('Tpip_out', np.nan),
                'Tpip_out2': row2.get('Tpip_out', np.nan),
                'Tbdy1': row1.get('Tbdy', np.nan),
                'Tbdy2': row2.get('Tbdy', np.nan),
                'Tid1': row1.get('Tid', np.nan),
                'Tid2': row2.get('Tid', np.nan),
                'Tod': row1.get('Tod', np.nan),  # 외부온도는 공통
                'Power1': row1.get('Power', np.nan),
                'Power2': row2.get('Power', np.nan),
            }
            
            # Action 변수들
            action = {
                'Tcon1': row1.get('Tcon', np.nan),
                'Tcon2': row2.get('Tcon', np.nan),
                'on_off1': 1 if row1.get('Tcon', 0) != 0 else 0,
                'on_off2': 1 if row2.get('Tcon', 0) != 0 else 0,
                'Ptarget': 0.0  # 나중에 설정할 목표 압력
            }
            
            # 통합
            combined_row = {**state, **action}
            matched_rows.append(combined_row)
        
        return pd.DataFrame(matched_rows)

class WorldModelVAE:
    """State 인코딩을 위한 VAE"""
    
    def __init__(self, state_dim=11, latent_dim=32):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.vae = None
        
    def build_model(self):
        """VAE 모델 구축"""
        # Encoder
        encoder_inputs = keras.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation='relu')(encoder_inputs)
        x = layers.Dense(32, activation='relu')(x)
        
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling)([z_mean, z_log_var])
        
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z])
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(32, activation='relu')(latent_inputs)
        x = layers.Dense(64, activation='relu')(x)
        decoder_outputs = layers.Dense(self.state_dim)(x)
        
        self.decoder = keras.Model(latent_inputs, decoder_outputs)
        
        # VAE
        vae_outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = keras.Model(encoder_inputs, vae_outputs)
        
        # Loss function
        def vae_loss(x, x_decoded):
            reconstruction_loss = keras.losses.mse(x, x_decoded)
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            return tf.reduce_mean(reconstruction_loss + kl_loss)
        
        self.vae.compile(optimizer='adam', loss=vae_loss)
        
    def train(self, X_train, epochs=100, batch_size=32):
        """VAE 훈련"""
        history = self.vae.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history

class WorldModelMDNRNN:
    """전이함수를 위한 MDN-RNN"""
    
    def __init__(self, latent_dim=32, action_dim=5, hidden_dim=256, n_mixtures=5):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_mixtures = n_mixtures
        self.model = None
        
    def build_model(self):
        """MDN-RNN 모델 구축"""
        # Input: current latent state + action
        inputs = keras.Input(shape=(None, self.latent_dim + self.action_dim))
        
        # LSTM layers
        lstm_out = layers.LSTM(
            self.hidden_dim, 
            return_sequences=True, 
            return_state=True
        )(inputs)
        
        lstm_output = lstm_out[0]
        
        # MDN outputs
        # Mixture weights
        pi = layers.Dense(self.n_mixtures, activation='softmax')(lstm_output)
        
        # Means for each mixture
        mu = layers.Dense(self.n_mixtures * self.latent_dim)(lstm_output)
        mu = layers.Reshape((-1, self.n_mixtures, self.latent_dim))(mu)
        
        # Standard deviations for each mixture  
        sigma = layers.Dense(self.n_mixtures * self.latent_dim, activation='softplus')(lstm_output)
        sigma = layers.Reshape((-1, self.n_mixtures, self.latent_dim))(sigma)
        
        self.model = keras.Model(inputs, [pi, mu, sigma])
        
    def mdn_loss(self, y_true, pi, mu, sigma):
        """MDN 손실 함수"""
        # y_true shape: (batch_size, seq_len, latent_dim)
        # pi shape: (batch_size, seq_len, n_mixtures)
        # mu shape: (batch_size, seq_len, n_mixtures, latent_dim)
        # sigma shape: (batch_size, seq_len, n_mixtures, latent_dim)
        
        y_true_expanded = tf.expand_dims(y_true, axis=2)  # (batch, seq, 1, latent_dim)
        
        # Gaussian probability
        diff = y_true_expanded - mu  # (batch, seq, n_mixtures, latent_dim)
        
        # Multivariate Gaussian (assuming diagonal covariance)
        exp_term = -0.5 * tf.reduce_sum(tf.square(diff) / tf.square(sigma), axis=-1)
        normalize_term = -0.5 * self.latent_dim * tf.math.log(2 * np.pi) - tf.reduce_sum(tf.math.log(sigma), axis=-1)
        
        gaussian_prob = tf.exp(exp_term + normalize_term)
        
        # Weighted mixture
        weighted_prob = pi * gaussian_prob
        mixture_prob = tf.reduce_sum(weighted_prob, axis=-1)
        
        # Negative log likelihood
        loss = -tf.math.log(mixture_prob + 1e-8)
        return tf.reduce_mean(loss)
    
    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """MDN-RNN 훈련"""
        def loss_fn(y_true, y_pred):
            pi, mu, sigma = y_pred
            return self.mdn_loss(y_true, pi, mu, sigma)
        
        self.model.compile(optimizer='adam', loss=loss_fn)
        
        # 더미 타겟 (실제로는 custom training loop 필요)
        dummy_target = np.zeros((len(X_train), X_train.shape[1], 3))
        
        history = self.model.fit(
            X_train, dummy_target,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history

class WorldModelController:
    """정책을 위한 Controller"""
    
    def __init__(self, latent_dim=32, action_dim=5, hidden_dim=64):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = None
        
    def build_model(self):
        """Controller 모델 구축"""
        inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(self.hidden_dim, activation='tanh')(inputs)
        x = layers.Dense(self.hidden_dim, activation='tanh')(x)
        
        # Continuous actions (Tcon1, Tcon2, Ptarget)
        continuous_actions = layers.Dense(3, activation='tanh')(x)
        
        # Discrete actions (on_off1, on_off2)
        discrete_actions = layers.Dense(2, activation='sigmoid')(x)
        
        # Combine actions
        outputs = layers.Concatenate()([continuous_actions, discrete_actions])
        
        self.model = keras.Model(inputs, outputs)
        
    def train_evolution_strategy(self, vae, mdn_rnn, reward_fn, generations=100):
        """Evolution Strategy로 Controller 훈련"""
        # ES 구현 (간단한 버전)
        population_size = 50
        noise_std = 0.1
        
        # 초기 가중치
        weights = self.model.get_weights()
        best_reward = float('-inf')
        best_weights = weights
        
        for generation in range(generations):
            population_rewards = []
            
            for individual in range(population_size):
                # 노이즈 추가
                noisy_weights = []
                for w in weights:
                    noise = np.random.normal(0, noise_std, w.shape)
                    noisy_weights.append(w + noise)
                
                self.model.set_weights(noisy_weights)
                
                # 환경에서 에피소드 실행하여 보상 계산
                reward = self._evaluate_policy(vae, mdn_rnn, reward_fn)
                population_rewards.append((reward, noisy_weights))
            
            # 상위 개체들 선택
            population_rewards.sort(key=lambda x: x[0], reverse=True)
            elite_size = population_size // 4
            elite_weights = [item[1] for item in population_rewards[:elite_size]]
            
            # 다음 세대 가중치 계산 (평균)
            new_weights = []
            for i, w in enumerate(weights):
                elite_w = np.array([elite[i] for elite in elite_weights])
                new_weights.append(np.mean(elite_w, axis=0))
            
            weights = new_weights
            
            # 최고 성능 업데이트
            current_best = population_rewards[0][0]
            if current_best > best_reward:
                best_reward = current_best
                best_weights = population_rewards[0][1]
            
            print(f"Generation {generation}: Best Reward = {current_best:.2f}")
        
        # 최고 가중치 설정
        self.model.set_weights(best_weights)
        return best_reward
    
    def _evaluate_policy(self, vae, mdn_rnn, reward_fn, episode_length=100):
        """정책 평가"""
        total_reward = 0
        
        # 초기 상태 (랜덤 또는 실제 데이터에서)
        current_state = np.random.normal(0, 1, (1, self.latent_dim))
        
        for step in range(episode_length):
            # 행동 선택
            action = self.model.predict(current_state, verbose=0)[0]
            
            # 다음 상태 예측 (MDN-RNN 사용)
            state_action = np.concatenate([current_state[0], action])
            state_action = state_action.reshape(1, 1, -1)
            
            pi, mu, sigma = mdn_rnn.model.predict(state_action, verbose=0)
            
            # 다음 상태 샘플링
            mixture_idx = np.random.choice(mdn_rnn.n_mixtures, p=pi[0, 0])
            next_state = np.random.normal(
                mu[0, 0, mixture_idx], 
                sigma[0, 0, mixture_idx]
            )
            
            # 보상 계산
            reward = reward_fn(current_state[0], action, next_state)
            total_reward += reward
            
            # 상태 업데이트
            current_state = next_state.reshape(1, -1)
        
        return total_reward

class WorldModelTrainer:
    """전체 World Models 훈련 파이프라인"""
    
    def __init__(self, data_path="../경남_사무실/LOG_SMARTCARE_*.csv"):
        self.data_processor = AirConditionerDataProcessor(data_path)
        self.vae = None
        self.mdn_rnn = None
        self.controller = None
        
    def train_full_pipeline(self):
        """전체 파이프라인 훈련"""
        print("1. 데이터 로딩 및 전처리...")
        data = self.data_processor.load_and_process_data()
        
        if data.empty:
            print("데이터가 없습니다. 파일 경로를 확인해주세요.")
            return
        
        # State와 Action 분리
        state_cols = ['Tpip_in1', 'Tpip_in2', 'Tpip_out1', 'Tpip_out2', 
                     'Tbdy1', 'Tbdy2', 'Tid1', 'Tid2', 'Tod', 'Power1', 'Power2']
        action_cols = ['Tcon1', 'Tcon2', 'on_off1', 'on_off2', 'Ptarget']
        
        # NaN 값 처리
        for col in state_cols + action_cols:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mean())
        
        states = data[state_cols].values
        actions = data[action_cols].values
        
        # 정규화
        state_scaler = StandardScaler()
        action_scaler = StandardScaler()
        
        states_normalized = state_scaler.fit_transform(states)
        actions_normalized = action_scaler.fit_transform(actions)
        
        print("2. VAE 훈련...")
        self.vae = WorldModelVAE(state_dim=len(state_cols))
        self.vae.build_model()
        self.vae.train(states_normalized, epochs=50)
        
        print("3. 잠재 표현 생성...")
        latent_states = self.vae.encoder.predict(states_normalized)[0]  # z_mean 사용
        
        print("4. MDN-RNN 훈련...")
        # 시퀀스 데이터 생성
        sequence_length = 10
        X_seq, y_seq = self._create_sequences(
            latent_states, actions_normalized, sequence_length
        )
        
        self.mdn_rnn = WorldModelMDNRNN(
            latent_dim=latent_states.shape[1], 
            action_dim=actions_normalized.shape[1]
        )
        self.mdn_rnn.build_model()
        self.mdn_rnn.train(X_seq, y_seq, epochs=50)
        
        print("5. Controller 훈련...")
        self.controller = WorldModelController(
            latent_dim=latent_states.shape[1],
            action_dim=actions_normalized.shape[1]
        )
        self.controller.build_model()
        
        # 보상 함수 정의 (예시)
        def reward_function(state, action, next_state):
            # 에너지 효율성과 온도 유지를 고려한 보상
            # 실제 구현시 도메인 지식에 따라 조정
            energy_penalty = -np.sum(action[3:5])  # on/off 페널티
            temp_reward = -np.abs(action[0] - 24) - np.abs(action[1] - 24)  # 목표온도 24도
            return energy_penalty + temp_reward
        
        best_reward = self.controller.train_evolution_strategy(
            self.vae, self.mdn_rnn, reward_function, generations=50
        )
        
        print(f"훈련 완료! 최고 보상: {best_reward:.2f}")
        
        # 모델 저장
        self.save_models()
        
    def _create_sequences(self, states, actions, sequence_length):
        """시퀀스 데이터 생성"""
        X, y = [], []
        
        for i in range(len(states) - sequence_length):
            # Input: state + action sequence
            state_action_seq = []
            for j in range(sequence_length):
                state_action = np.concatenate([states[i+j], actions[i+j]])
                state_action_seq.append(state_action)
            
            X.append(state_action_seq)
            
            # Target: next state sequence
            next_states = states[i+1:i+sequence_length+1]
            y.append(next_states)
        
        return np.array(X), np.array(y)
    
    def save_models(self):
        """모델 저장"""
        os.makedirs('world_models_checkpoints', exist_ok=True)
        
        if self.vae and self.vae.vae:
            self.vae.vae.save('world_models_checkpoints/vae_model.h5')
            print("VAE 모델 저장됨: world_models_checkpoints/vae_model.h5")
            
        if self.mdn_rnn and self.mdn_rnn.model:
            self.mdn_rnn.model.save('world_models_checkpoints/mdn_rnn_model.h5')
            print("MDN-RNN 모델 저장됨: world_models_checkpoints/mdn_rnn_model.h5")
            
        if self.controller and self.controller.model:
            self.controller.model.save('world_models_checkpoints/controller_model.h5')
            print("Controller 모델 저장됨: world_models_checkpoints/controller_model.h5")

def main():
    """메인 실행 함수"""
    print("=== World Models for Air Conditioner Control ===")
    
    # 훈련 실행
    trainer = WorldModelTrainer()
    trainer.train_full_pipeline()
    
    print("\n훈련이 완료되었습니다!")
    print("저장된 모델들:")
    print("- world_models_checkpoints/vae_model.h5")
    print("- world_models_checkpoints/mdn_rnn_model.h5") 
    print("- world_models_checkpoints/controller_model.h5")

if __name__ == "__main__":
    main()