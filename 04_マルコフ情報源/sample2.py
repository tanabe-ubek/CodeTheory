import numpy as np

# 遷移確率行列
P = np.array([
    [0.4, 0.6],
    [0.2, 0.8]
])

# ステップ数
T = 10

print("P^t:")
for t in range(1, T + 1):
    Pt = np.linalg.matrix_power(P, t)
    print(f"\nP^{t} =")
    print(np.round(Pt, 6))  # 小数を見やすく丸めて表示
