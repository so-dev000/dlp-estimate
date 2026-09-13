# DLP estimate

有限体上の離散対数問題（DLP）の資源見積りを行うためのプロジェクト。

## 環境構築

```bash
uv sync --locked
```

## フォルダ構成

```text
dlp-estimate/
├── src/
│   ├── __init__.py
│   ├── field.py          # 有限体演算・符号化・DLP 入力の検証
│   ├── arithmetic.py     # Qualtran による有限体演算回路
│   └── classical/
│       ├── __init__.py
│       ├── pohling_hellman.py  # Pohlig–Hellman 法
│       └── pollard_rho.py      # Pollard rho 法
└── tests/ # リグレッション防止目的 (一旦Codexで作成・要確認)
    ├── __init__.py
    ├── test_field.py
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
