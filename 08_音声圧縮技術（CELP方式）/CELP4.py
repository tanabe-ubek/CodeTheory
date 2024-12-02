import numpy as np
from scipy.signal import lfilter, lfilter_zi
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

# サンプルデータの生成（正弦波 + ノイズ）
fs = 8000  # サンプリング周波数
t = np.linspace(0, 1, fs)
signal = np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))

# パラメータ
frame_size = 80  # フレーム長（10ms）
overlap = 40  # フレームオーバーラップ（50%）
lpc_order = 16  # LPC係数の次数（精度向上）
gamma1 = 0.9  # 感覚重み付けフィルタパラメータ
gamma2 = 0.6

# フレーム分割（オーバーラップあり）
def frame_signal_with_overlap(signal, frame_size, overlap):
    step = frame_size - overlap
    num_frames = (len(signal) - overlap) // step
    frames = np.array([signal[i*step:i*step+frame_size] for i in range(num_frames)])
    return frames

frames = frame_signal_with_overlap(signal, frame_size, overlap)

# LPC係数の計算
def compute_lpc(frame, order):
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # 自己相関
    r = autocorr[:order+1]
    R = toeplitz(r[:-1])  # Toeplitz行列
    a = np.linalg.solve(R, -r[1:])  # LPC係数
    error = r[0] + np.dot(r[1:], a)  # 残差エネルギー
    return np.concatenate(([1], a)), error

# 感覚重み付けフィルタの適用
def perceptual_filter(lpc_coeffs, gamma1, gamma2):
    weighted_coeffs = [(1 - gamma1) * coeff for coeff in lpc_coeffs]
    return [1] + weighted_coeffs

# スパースコードブックの生成
def generate_sparse_codebook(size, length, non_zero_count):
    codebook = []
    for _ in range(size):
        vector = np.zeros(length)
        positions = np.random.choice(length, non_zero_count, replace=False)  # 非ゼロ要素の位置をランダムに選択
        signs = np.random.choice([-1, 1], non_zero_count)  # 非ゼロ要素の符号をランダムに選択
        for pos, sign in zip(positions, signs):
            vector[pos] = sign
        codebook.append(vector)
    return np.array(codebook)

# フレーム全体にスパースベクトルを周期的に展開
def expand_to_frame(sparse_vector, frame_size):
    repeated_vector = np.tile(sparse_vector, (frame_size // len(sparse_vector) + 1))  # 繰り返し
    return repeated_vector[:frame_size]  # フレームサイズに切り詰める

# フィルタの初期条件を設定
def filter_with_initial_conditions(lpc_coeffs, frame, zi=None):
    if zi is None:
        zi = lfilter_zi([1], lpc_coeffs) * frame[0]  # フレームの最初のサンプルに基づいて設定
    filtered, zf = lfilter(lpc_coeffs, [1], frame, zi=zi)  # フィルタ処理
    return filtered, zf

# スパースコードブックの生成
codebook_size = 512  # コードブックサイズを拡大
codebook_length = 17  # スパースベクトルの長さ
non_zero_count = 4  # 非ゼロ要素の数
sparse_codebook = generate_sparse_codebook(codebook_size, codebook_length, non_zero_count)

# CELPコーディング
zi = None  # 初期条件
reconstructed_signal = np.zeros(len(signal))  # 再合成信号
step = frame_size - overlap

# フレームごとの符号化と再合成
for i, frame_start in enumerate(range(0, len(signal) - frame_size, step)):
    frame = signal[frame_start:frame_start + frame_size]

    # LPC解析
    lpc_coeffs, _ = compute_lpc(frame, lpc_order)
    weighted_coeffs = perceptual_filter(lpc_coeffs, gamma1, gamma2)

    # 残差計算
    residual, zi = filter_with_initial_conditions(lpc_coeffs, frame, zi)

    # コードブック検索
    min_error = float('inf')
    best_excitation = None

    for sparse_vector in sparse_codebook:
        expanded_vector = expand_to_frame(sparse_vector, frame_size)  # フレーム全体に展開
        synthesized = lfilter(weighted_coeffs, [1], expanded_vector)  # フィルタを通して合成信号を生成
        error = np.sum((residual - synthesized) ** 2)  # 誤差エネルギー
        if error < min_error:
            min_error = error
            best_excitation = expanded_vector

    # 再合成
    reconstructed_frame = lfilter([1], lpc_coeffs, best_excitation)
    reconstructed_signal[frame_start:frame_start + frame_size] += reconstructed_frame  # フレームを加算
    zi = None  # フレーム切り替え時に初期条件をリセット


# フレーム単位でプロット（例：最初のフレーム）
plt.figure(figsize=(12, 4))
plt.title("First Frame (80 samples)")
plt.plot(signal[:80], label="Original Signal")
plt.plot(reconstructed_signal[:80], label="Reconstructed Signal")
plt.legend()
plt.show()


# 結果をプロット
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.title("Original Signal")
plt.plot(signal)
plt.subplot(3, 1, 2)
plt.title("Reconstructed Signal (Improved)")
plt.plot(reconstructed_signal)
plt.subplot(3, 1, 3)
plt.title("Residual Difference (Original - Reconstructed)")
plt.plot(signal - reconstructed_signal)
plt.tight_layout()
plt.show()
