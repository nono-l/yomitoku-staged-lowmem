# 次のセッションへ

このファイルは、何も知らない AI / 人がこの部品を壊さずに続けるためのメモ。
コメント方針は [CODING.md](CODING.md)。

このセッション（2026-09-20）: P1a / C2 まで緑。
次は P1b（検出だけ ONNX。既定の認識は旧面）。

---

## これは何か

画素から図中の活字を、箱と信頼度と読み順つきで取り出す部品。分野は知らない。

リポジトリ: https://github.com/nono-l/yomitoku-staged-lowmem

---

## 絶対に戻さないこと

1. **検出と認識を同じプロセスに載せない。**
2. **分野語をスキーマに入れない。**
3. **DBNet / PARSeq を今夜書き直さない。**
4. **実行のたびに Hugging Face の latest を黙って引かない。** revision は `weights/manifest.json`。
5. **提出してから動かす、にしない。**
6. **safetensors を git に載せない。** 別置きは `weights/pinned/`。

---

## 今動いている面

- `python3 tests/test_assemble.py` … モデル無し、本文残存
- `python3 tests/test_weights.py` … 無い・違うと赤。実体があればハッシュも見る
- `run_staged.sh` は検出前に `weights/verify_weights.py`
- ローダはまだ YomiToku。ピンは「何を使ってよいか」の台帳

ピン済（2026-09-20）:
- detector `3ac8375d1fd124e8074b013a32f2085d1199654c` sha256 fa090966… 102492968 bytes
- recognizer `abd24b78292afa849e206b42ebf72464ffd7eae5` sha256 498db515… 36097960 bytes

---

## 互い違いの順

| 順 | ID | 状態 |
|---|---|---|
| 1-6 | C0–T1 | 済 |
| 7 | P1a | 済 manifest + pin |
| 8 | C2 | 済 verify が入口の前 |
| 9 | P1b | 次。検出だけ ONNX |
| 10 | T2 | 旧検出と ONNX 検出 |
| 11+ | C3–P2 | 未 |

---

## 既知の粗

- YomiToku の from_pretrained はまだ HF キャッシュを見る。ピン目録と実際ローダの結合は P1b の前に処理するか、ONNX 化で踏まなくなる。
- サンドボックスは RAM 1.9GiB。
