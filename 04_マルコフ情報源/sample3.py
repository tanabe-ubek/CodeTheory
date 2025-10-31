import numpy as np

# 遷移確率行列（S0, S1, S2 の順）
P = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.4, 0.6, 0.0]
])

# 初期分布（例：すべてS0にいる場合）
omega0 = np.array([1.0, 0.0, 0.0])

# ステップ数
T = 10

print("t | ω0        ω1        ω2")
print("-------------------------------------")
omega = omega0.copy()
for t in range(T + 1):
    print(f"{t:>1} | {omega[0]:.6f}  {omega[1]:.6f}  {omega[2]:.6f}")
    omega = omega @ P  # 次の時刻へ更新

# P^t の出力（任意）
print("\nP^t:")
for t in range(1, T + 1):
    Pt = np.linalg.matrix_power(P, t)
    print(f"\nP^{t} =\n{np.round(Pt, 6)}")
