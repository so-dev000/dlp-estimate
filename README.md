# DLP estimate

有限体上の離散対数問題（DLP）の資源見積りを行うためのプロジェクト。

## 環境構築

```bash
uv sync --locked
```

## スクリプト実行

```bash
uv run python script/sweep.py  # 複数入力の資源見積り
uv run python script/show_call_graph.py  # call-graph 表示
```

## フォルダ構成

```text
dlp-estimate/
├── script/
│   ├── sweep.py             # 複数入力の資源見積り
│   └── show_call_graph.py   # call-graph表示
├── src/
│   ├── __init__.py
│   ├── field.py             # 有限体演算・符号化・DLP 入力の検証
│   ├── arithmetic.py        # Qualtran による有限体演算回路
│   ├── shor.py              # Shor-DLP
│   ├── logical_resources.py # 論理リソース見積り
│   ├── search.py            # parameter sweep
│   └── classical/
│       ├── __init__.py
│       ├── pohling_hellman.py  # Pohlig–Hellman 法
│       └── pollard_rho.py      # Pollard rho 法
├── results/
│   └── sweep_toy/           # sweep.py の出力 (rows.json・rows.csv)
└── tests/ # リグレッション防止目的 (一旦Codexで作成・要確認)
    ├── __init__.py
    ├── test_field.py
    ├── test_arithmetic.py
    ├── test_shor.py
    └── classical/
        ├── __init__.py
        ├── test_pohling_hellman.py
        └── test_pollard_rho.py
```

## テスト実行

```bash
uv run pytest
```

## 参考

- [Qualtran](https://qualtran.readthedocs.io/en/latest/index.html)
- [galois](https://mhostetter.github.io/galois/latest/)
