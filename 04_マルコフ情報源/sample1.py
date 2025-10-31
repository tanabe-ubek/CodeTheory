import numpy as np

# 遷移確率行列 P
# [ [S0→S0, S1→S0],
#   [S0→S1, S1→S1] ] ではなく、
# 行列の定義として： [現在の状態]×[次の状態]
P = np.array([
    [0.4, 0.6],  # S0→S0, S0→S1
    [0.2, 0.8]   # S1→S0, S1→S1
])

# 初期分布（例：S0にいる確率=1.0）
omega = np.array([1.0, 0.0])  # ω^0 = [ω_0^0, ω_1^0]

# ステップ数
T = 10

print("t | ω0        ω1")
print("-----------------------")
for t in range(T + 1):
    print(f"{t:>1} | {omega[0]:.6f}  {omega[1]:.6f}")
    omega = omega @ P  # 次の時刻へ更新
