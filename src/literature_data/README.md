# 文献由来 DLP インスタンス

このディレクトリには、有限体上の離散対数問題（DLP）について、
文献で実際に計算されたインスタンスを、本研究の量子リソース見積で利用できる
単一多項式基底の形式に変換したデータを保存する。

生成済み JSON は `script/generate_literature_instances.py` によって
文献中の定義から再生成・検証できる。

## DGP21

出典:

De Micheli, Gaudry, Pierrot,
"Lattice Enumeration for Tower NFS: a 521-bit Discrete Logarithm Computation",
ASIACRYPT 2021, ePrint 2021/707.

対象は約 521-bit の `F_{p^6}` 上の DLP である。

元論文では

\[
\mathbb{F}_{p^3}
=

\mathbb{F}_p[t]/(t^3-t+1),
\]

\[
\mathbb{F}_{p^6}
=

\mathbb{F}_{p^3}[x]/(x^2+cx+1)
\]

という tower 表現を使用する。

本研究では

\[
X=t+x
\]

と置き、resultant によって `t` を消去することで

\[
\mathbb{F}_{p^6}
\simeq
\mathbb{F}_p[X]/(f(X))
\]

という単一多項式表現へ変換する。

さらに

\[
q=p^2-p+1
\]

を位数とする部分群へ

\[
C=\frac{p^6-1}{q}
\]

を用いて射影し、

\[
g=X^C,\qquad
h=\mathrm{target}^C
\]

を canonical DLP instance とする。

生成スクリプトでは、論文に掲載された既知の離散対数を用いて

\[
g^{\log_g h}=h
\]

を確認する。

`dgp21.json` はこの変換後の `p, r, f, q, g, h` を保存する。

## GKLWZ21

出典:

Granger, Kleinjung, Lenstra, Wesolowski, Zumbrägel,
"Computation of a 30750-Bit Binary Field Discrete Logarithm",
ePrint 2020/965, Mathematics of Computation 2021.

対象は

\[
\mathbb{F}_{2^{30750}}
\]

上の DLP である。

元論文では

\[
\mathbb{F}_{2^{30}}
=

\mathbb{F}_2[t]/(t^{30}+t+1),
\]

\[
\mathbb{F}_{2^{30750}}
=

\mathbb{F}_{2^{30}}[x]/(x^{1025}+x+t^3)
\]

という tower 表現を使用する。

本研究では

\[
X=x,\qquad
u=X^{1025}+X=t^3
\]

と置く。

`t^{30}+t+1=0` より

\[
t=u^{10}+1
\]

であるため、

\[
u^{30}+u^{20}+u^{10}+u+1=0
\]

が得られる。

これに

\[
u=X^{1025}+X
\]

を代入して、次数 30750 の単一多項式 `f(X)` を構成する。

論文の generator

\[
g=x+t^9
\]

も同じ基底へ変換する。

target は論文 Appendix B の \(\pi\) の二進展開から定義される要素を
同じ方法で生成する。

生成された target について、以下の SHA-256 を検証する。

```text
ec6f487c2a2b9b6aba259d2851aae888c127ea386bd6e9e584e81bcfbab14c9d
```

## KLEINJUNG14

出典:

Thorsten Kleinjung,
"Discrete Logarithms in GF(2^1279)",
NMBRTHRY mailing list, 17 Oct 2014.

対象は

\[
\mathbb{F}_{2^{1279}}
\]

上の DLP である。

元論文では \(q=2^8\) とし、\(h_0=x^5+x^3+x\)、\(h_1=x+1\) を用いて

\[
f=(h_1(x^q)x+h_0(x^q))/x
=x^{1279}+x^{767}+x^{256}+x^{255}+1
\]

という単一多項式で体を定義する。変換は不要であり、
`kleinjung14.json` にはこの5項式をそのまま保存する。

\(2^{1279}-1\) は素数のため、\(x\) は乗法群全体の生成元であり、
\(q\) の完全因数分解が既知である。古典側は FFS による関係収集と
Granger–Kleinjung–Zumbrägel の powers-of-2 descent
(quasi-polynomial descent) による降下で計算され、
合計4コア年未満で離散対数が求められた。
GKLWZ21 (\(r=30750\)) と異なり \(r=1279\) のため量子リソース見積も実行可能である。

target は論文付属の Magma スクリプトと同様に
\(\pi\) の二進展開から定義される要素であり、
論文が掲載する既知の離散対数

\[
x^{\log}=t
\]

を生成スクリプトで確認する。
生成された target について、以下の SHA-256 を検証する。

```text
d0302169937c265a43dc3be330369ffafa9a865edddf477a1ea2546a23c640d1
```

`kleinjung14.json` は `p, r, f, q, g, h` と既知対数を保存する。
