# 次のセッションへ

2026-09-20: T5。encoder/decoder を幅動的にして既定 rec と比べた。C4 しない。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. dynw 二グラフを `run_staged.sh` の既定にしない。
5. 実体を git に載せない。

---

## 今動いている面

- 検出既定: 別置き ONNX
- 認識既定: torch tiny dynamic_width
- T5: dynw 二グラフ vs 既定 rec で 36/37、本文 3 つ残る
- 差分 1 件: `。」` vs `。丼`（本文ではない）
- 先回の 1000/1200 は 3px 箱の雑音。幅動的経路では消えた

encoder_dynw sha256 `1baa7e40…`  Drive `1sDOBIYnkAWFDlaydO6k3rmgiwaCWCJ5s`
decoder_step_dynw sha256 `e00d9678…` Drive `1J8G8suiR57lSgfJsR3FUZ0KYM89PVs-r`

---

## 次

C4 はまだしない。`。」` / `。丼` を許すなら比較用のまま既定に近づける。
切るなら `recognize_two_graph.py` の既定ファイルを dynw にし、`run_staged.sh` はまだ触らない。
