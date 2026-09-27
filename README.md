# DLP estimate

有限体上の離散対数問題（DLP）の資源見積りを行うためのプロジェクト。

## 環境構築

```bash
uv sync --locked
```

## スクリプト実行

```bash
uv run python script/estimate_qualtran.py  # 複数入力の資源見積り
uv run python script/show_call_graph.py  # call-graph 表示
```

## フォルダ構成

```text
dlp-estimate/
├── script/
│   ├── estimate_qualtran.py             # 複数入力の資源見積り
│   └── show_call_graph.py   # call-graph表示
├── src/
│   ├── __init__.py
│   ├── field.py             # 有限体演算・符号化・DLP 入力の検証
│   ├── arithmetic.py        # Qualtran による有限体演算回路
│   ├── exponentiation.py    # 指数演算 Bloq
│   ├── oracle.py            # DLP oracle Bloq
│   ├── shor.py              # Shor-DLP 本体・設定・成功率
│   ├── generate_graph.py
│   ├── resource_estimate/   # 資源評価
│   │   ├── __init__.py
│   │   ├── logical.py           # 論理計数・回転合成・魔状態需要
│   │   ├── physical.py          # 物理コストモデル評価 (構成は固定指定)
│   │   ├── configuration.py     # 指定 factory の蒸留距離・data 構成の探索
│   │   └── search.py            # 1点評価・parameter sweep
└── results/               # 生成物
    └── qualtran/              # estimate_qualtran.py の出力 (config.txt・metadata.json・rows.json・rows.csv・*.png)
```

## 参考

- [Qualtran](https://qualtran.readthedocs.io/en/latest/index.html)
