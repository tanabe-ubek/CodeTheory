Enhanced by ChatGPT (Information Theory Expert)
https://chatgpt.com/share/67484265-ec54-800f-b553-d3738e985c9c


```sh
pip install numpy scipy matplotlib
```

# CELP (Code-Excited Linear Prediction) 
非可逆な音声圧縮技術
## アルゴリズム概要
1. 音声信号の分解（LPC分析）
2. 励振信号の選択（コードブック検索）
3. 合成フィルタを用いた再合成
4. 誤差計算と最適なパラメータの選択
5. 量子化とビットストリーム生成

## 1.LPC分析
時刻nのデータx[n]と時刻n-iのデータx[n-i]との相関を調べ、
$$ x[n] \approx  \Sigma_{i=1}^{p} a_i x[n-i] $$
に最適となる$a_i$を求める。

## 2. 励振信号の選択（コードブック検索）
LPCで近似できた値とx[n]との値の差（残差）をe[n]とし、このe[n]をさらに近似する。
コードブックiは各時刻nに対する値$e_i[n]$を持つ。
- 適応コードブック：周期的な成分を模倣するデータ
$$ e_{adaptive}[n]=g⋅e_{prev}[n−r] $$
- 固定コードブック：ランダム成分やモデル成分を模倣するデータ
 1. 長さ17のビット列a[i]を作る。ただし、0が13個、-1または+1が4個
 2.  時刻 17m+iにa[i]の雑音が起こる。ただし、その時刻だけ±1ではなく、そこをピークとする山のイメージ。

 これらの候補の中から、最も残差e[n]を近似できるコードブックを選ぶ。

## 各種パラメータ
CS-ACELP (Conjugate-Structure Algebraic Code-Excited Linear Prediction)、特にITU-T G.729規格のパラメータ

- LPCフィルタの次数 p: 10
- フレーム長: 10ms（80サンプル）
- LPC係数更新: フレームごと（10msごとに再計算）
LPC係数は分析窓（通常20ms）を使用して計算され、隣接フレーム間で線形補間されます。

適応コードブック

遅延範囲 
𝜏
τ: 20 ～ 143サンプル（約2.5ms ～ 17.875ms）
ピッチ周期に対応する遅延を探索します。
精密な遅延値:
適応コードブックの遅延値は、1サンプル精度（整数部）に加えて、小数部（1/3サンプル精度）も考慮します。これにより、より正確なピッチ成分を表現可能です。
ゲイン 
𝑔
g: 3ビットで量子化（8段階）
適応コードブック信号の強度を調整するためのゲイン値。
固定コードブック

コードブックのサイズ: 17ビットインデックスで符号化（65,536通りの候補）
アルジェブラ的構造を持つスパースベクトル集合を使用し、効率的に表現。
ベクトル構造:
固定コードブック信号はスパースベクトル（1フレームあたり4つの非ゼロパルス）で構成されます。
ゲイン 
𝑔
g: 5ビットで量子化（32段階）
各固定コードブック候補に対するゲインを量子化します。

