import numpy as np
from scipy.signal import lfilter
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

# サンプルデータの生成（正弦波 + ノイズ）
fs = 8000  # サンプリング周波数
t = np.linspace(0, 1, fs)
signal = np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))

# パラメータ
frame_size = 160  # フレーム長（20ms）
lpc_order = 10  # LPC係数の次数

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

# コードブックの生成
def generate_codebook(size, length):
    return np.random.randn(size, length)  # ランダムコードブック

codebook_size = 64
codebook = generate_codebook(codebook_size, frame_size)

# CELPコーディング（単一フレームの例）
frame = frames[0]  # 最初のフレームを選択

# LPC解析
lpc_coeffs, _ = compute_lpc(frame, lpc_order)

# LPCフィルタによる残差計算
residual = lfilter(lpc_coeffs, [1], frame)

# コードブック検索
best_index = None
min_error = float('inf')
best_excitation = None

for i, excitation in enumerate(codebook):
    # フィルタを通して合成信号を生成
    synthesized = lfilter([1], lpc_coeffs, excitation)
    error = np.sum((residual - synthesized) ** 2)  # 誤差エネルギー
    if error < min_error:
        min_error = error
        best_index = i
        best_excitation = excitation

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
plt.title("Reconstructed Signal")
plt.plot(reconstructed)
plt.tight_layout()
plt.show()
