import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import japanize_matplotlib

# ----------------------------------------------------
# 1. モデルとデータの初期設定
# ----------------------------------------------------

# 状態空間モデルの次元設定
STATE_DIM = 2  # 状態ベクトル（例：[x, y]）の次元
OBS_DIM = 2  # 観測ベクトル（例：[x_obs, y_obs]）の次元
DATA_POINTS = 100
EM_UPDATE_FREQ = 20  # EMアルゴリズムでパラメータを修正する頻度（例：20ステップごと）

# 初期パラメータ（適当な初期値）
initial_state_mean = np.zeros(STATE_DIM)
initial_state_covariance = np.eye(STATE_DIM) * 1.0
transition_matrix = np.eye(STATE_DIM) * 1.01  # 少しずつ変化するモデル
observation_matrix = np.eye(OBS_DIM)
process_noise_covariance = np.eye(STATE_DIM) * 0.1
measurement_noise_covariance = np.eye(OBS_DIM) * 1.0

# ----------------------------------------------------
# 2. カルマンフィルタの初期化
# ----------------------------------------------------

# 初期パラメータでKalmanFilterインスタンスを作成
kf = KalmanFilter(
    transition_matrices=transition_matrix,
    observation_matrices=observation_matrix,
    transition_covariance=process_noise_covariance,
    observation_covariance=measurement_noise_covariance,
    initial_state_mean=initial_state_mean,
    initial_state_covariance=initial_state_covariance,
)

# ----------------------------------------------------
# 3. テストデータの生成
# ----------------------------------------------------
# 簡易なシミュレーションデータ（ノイズが乗ったサインカーブ）
time = np.arange(DATA_POINTS)
# 真の状態（ここでは観測ノイズ前の理想的なデータ）
true_states = np.vstack(
    [
        2 * np.sin(time * 0.2) + time * 0.1,  # x座標
        2 * np.cos(time * 0.1) - time * 0.05,  # y座標
    ]
).T
# 観測データ（真の状態にノイズを加えたもの）
measurements = true_states + np.random.normal(0, 0.15, size=(DATA_POINTS, OBS_DIM))

# ----------------------------------------------------
# 4. 逐次処理ループ (予測、修正、パラメータ修正)
# ----------------------------------------------------

# 逐次処理のための初期化
filtered_state_means = []
predicted_next_means = []

# 状態の初期推定値と共分散
current_state_mean = initial_state_mean
current_state_covariance = initial_state_covariance

# EM学習のために観測データを格納するリスト
em_data_buffer = []

print("--- 逐次カルマンフィルタリングとEM学習の開始 ---")

for k in range(DATA_POINTS):
    current_measurement = measurements[k]
    em_data_buffer.append(current_measurement)

    # ---------------------------
    # A. 次の値の予測 (Predict)
    # ---------------------------
    # **kf.predict()** を使って、観測値 y_k が来る前に次の状態 x_{k+1} を予測します。
    # ここでは、観測値 y_k の予測値 y_{k|k-1} ではなく、状態 x_k の予測値を求めます。

    # 時点 k のデータ y_k を処理する前に、次の状態 x_{k+1} を予測したい場合
    # kf.transition_matrices は状態遷移を表すので、それを適用します。

    # 実際には、filter_updateの最初のステップが予測ステップですが、
    # 予測値（Predictive Mean）を明示的に取得するために、ここでは次の時点の予測計算をします。

    # **【注】次の値の予測 (Prediction/Forecasting)**
    # k+1 時点の状態の予測（観測 y_k を用いる前の予測）
    # $\hat{x}_{k+1|k} = A_k \hat{x}_{k|k}$
    predicted_state_mean = kf.transition_matrices @ current_state_mean
    predicted_next_means.append(predicted_state_mean)

    # ---------------------------
    # B. 状態の修正 (Filter Update)
    # ---------------------------
    # 観測値 y_k を与えて、状態 $\hat{x}_{k|k}$ を修正します。
    # filter_update は「予測ステップ」と「修正ステップ」を内部で実行し、
    # 最終的に修正後の状態 $\hat{x}_{k|k}$ と共分散 $\mathbf{P}_{k|k}$ を返します。

    new_state_mean, new_state_covariance = kf.filter_update(
        current_state_mean, current_state_covariance, observation=current_measurement
    )

    # 結果を保存
    filtered_state_means.append(new_state_mean)

    # 次のループのために状態を更新
    current_state_mean = new_state_mean
    current_state_covariance = new_state_covariance

    # ---------------------------
    # C. パラメータ修正 (EM Algorithm)
    # ---------------------------
    if (k + 1) % EM_UPDATE_FREQ == 0 and k > 0:
        print(f"--- ステップ {k+1}: EMアルゴリズムでパラメータを修正 ---")

        # 観測データをNumpy配列に変換
        observations_to_fit = np.array(em_data_buffer)

        # EMアルゴリズムで最適なパラメータを推定
        # n_iter は反復回数
        kf = kf.em(observations_to_fit, n_iter=5)

        # パラメータが更新されたことを確認（例：遷移行列）
        print(f"  > 新しい遷移行列:\n{kf.transition_matrices}")

        # バッファをクリア（またはスライド）
        # 例として、ここではバッファ全体をリセット
        em_data_buffer = []

# リストをNumpy配列に変換
filtered_state_means = np.array(filtered_state_means)
# 予測はデータ数より1つ多く計算されるため、最後の1つを削除
predicted_next_means = np.array(predicted_next_means)[:-1]


# ----------------------------------------------------
# 5. 結果の可視化
# ----------------------------------------------------

plt.figure(figsize=(12, 6))

# X座標のプロット
plt.subplot(1, 2, 1)
plt.plot(time, true_states[:, 0], "g-", label="真の状態 (True State X)")
plt.plot(time, measurements[:, 0], "r.", label="観測データ (Measurement X)")
plt.plot(
    time, filtered_state_means[:, 0], "b-", label="フィルタリング結果 (Filtered X)"
)
plt.plot(time[1:], predicted_next_means[:, 0], "y--", alpha=0.6, label="次の値の予測 X")
plt.title("X座標の推定と予測")
plt.xlabel("ステップ")
plt.ylabel("値")
plt.legend()
plt.grid(True)

# Y座標のプロット
plt.subplot(1, 2, 2)
plt.plot(time, true_states[:, 1], "g-", label="真の状態 (True State Y)")
plt.plot(time, measurements[:, 1], "r.", label="観測データ (Measurement Y)")
plt.plot(
    time, filtered_state_means[:, 1], "b-", label="フィルタリング結果 (Filtered Y)"
)
plt.plot(time[1:], predicted_next_means[:, 1], "y--", alpha=0.6, label="次の値の予測 Y")
plt.title("Y座標の推定と予測")
plt.xlabel("ステップ")
plt.ylabel("値")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 最終的なパラメータの確認
print("\n--- 最終的なEM学習後のパラメータ ---")
print("状態遷移行列 (A):\n", kf.transition_matrices)
print("プロセスノイズ共分散 (Q):\n", kf.transition_covariance)
print("観測ノイズ共分散 (R):\n", kf.observation_covariance)
