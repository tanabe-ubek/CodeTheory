import numpy as np

# === 遷移確率行列（S0, S1, S2） ===
P = np.array([
    [0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2],
    [0.4, 0.6, 0.0]
])

# 最大ステップ数
T = 10

print("=== P^t の一覧 ===")
for t in range(1, T + 1):
    Pt = np.linalg.matrix_power(P, t)
    print(f"\nP^{t} =")
    print(np.round(Pt, 6))
