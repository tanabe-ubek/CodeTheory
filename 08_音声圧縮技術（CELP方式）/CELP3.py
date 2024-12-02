import numpy as np
from scipy.signal import lfilter
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

# サンプルデータの生成（正弦波 + ノイズ）
fs = 8000  # サンプリング周波数
t = np.linspace(0, 1, fs)
signal = np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))

# パラメータ
frame_size = 80  # フレーム長（10ms）
lpc_order = 16  # LPC係数の次数（精度向上）
gamma1 = 0.9  # 感覚重み付けフィルタパラメータ
gamma2 = 0.6

# フレーム分割
def frame_signal(signal, frame_size):
    num_frames = len(signal) // frame_size
    frames = signal[:num_frames * frame_size].reshape(num_frames, frame_size)
    return frames

frames = frame_signal(signal, frame_size)

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

# スパースコードブックの生成
codebook_size = 512  # コードブックサイズを拡大
codebook_length = 17  # スパースベクトルの長さ
non_zero_count = 4  # 非ゼロ要素の数
sparse_codebook = generate_sparse_codebook(codebook_size, codebook_length, non_zero_count)

# CELPコーディング（1フレームの例）
frame = frames[0]  # 最初のフレームを選択

# LPC解析
lpc_coeffs, _ = compute_lpc(frame, lpc_order)
weighted_coeffs = perceptual_filter(lpc_coeffs, gamma1, gamma2)

# LPCフィルタによる残差計算
residual = lfilter(lpc_coeffs, [1], frame)

# コードブック検索
min_error = float('inf')
best_index = None
best_excitation = None

# コードブック検索と最適励振信号の決定
for i, sparse_vector in enumerate(sparse_codebook):
    expanded_vector = expand_to_frame(sparse_vector, frame_size)  # フレーム全体に展開
    synthesized = lfilter(weighted_coeffs, [1], expanded_vector)  # フィルタを通して合成信号を生成
    error = np.sum((residual - synthesized) ** 2)  # 誤差エネルギー
    if error < min_error:
        min_error = error
        best_index = i
        best_excitation = expanded_vector

# 再合成信号
reconstructed = lfilter([1], lpc_coeffs, best_excitation)

# 結果をプロット
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.title("Original Frame")
plt.plot(frame)
plt.subplot(3, 1, 2)
plt.title("Residual Signal")
plt.plot(residual)
plt.subplot(3, 1, 3)
plt.title("Reconstructed Signal (Improved)")
plt.plot(reconstructed)
plt.tight_layout()
plt.show()

