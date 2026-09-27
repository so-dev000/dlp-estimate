# DLP estimate

`GF(p^r)` 上の離散対数問題（DLP）に対する Shor 型アルゴリズムの、Qualtran ベースの資源見積り。

## 環境構築

```bash
uv sync --locked
```

## 使い方

### 資源見積り

```bash
uv run python script/estimate_qualtran.py
```

## フォルダ構成

```text
dlp-estimate/
├── script/
│   ├── estimate_qualtran.py          # 資源見積りのエントリポイント
│   └──generate_literature_instances.py  # 文献インスタンス JSON の再生成・検証
├── src/
│   ├── field.py                      # GF(p^r) の仕様・演算・DLP 入力検証
│   ├── arithmetic.py                 # 有限体演算 Bloq（加算・定数乗算など）
│   ├── exponentiation.py             # 指数演算 Bloq
│   ├── oracle.py                     # DLP oracle Bloq（h^a g^b）
│   ├── shor.py                       # Shor-DLP 本体（H → oracle → 逆QFT）
│   ├── validation.py                 # 確率・入力値の検証ヘルパー
│   ├── literature_instances.py       # literature_data JSON → Params 変換・点集合定義
│   ├── literature_data/              # 文献由来インスタンスの canonical JSON（dgp21 等）+ README
│   ├── generate_graph.py             # results グラフ生成（scaling_qubits / scaling_runtime_total）
│   └── resource_estimate/
│       ├── logical.py                # 論理計数・回転合成・magic state
│       ├── physical.py               # Qualtran PhysicalCostModel による単発 FT 評価
│       ├── configuration.py          # factory 蒸留距離・data 距離・data block の探索
│       ├── success.py                # Mosca 下限・実装誤差込み下限・反復回数
│       └── search.py                 # Params → EvalRow の 1 点評価・sweep
└── results/
```

## 参考

- [Qualtran](https://qualtran.readthedocs.io/en/latest/index.html)
