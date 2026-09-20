# 次のセッションへ

2026-09-20: 認識の Dataset / Tokenizer を剥いだ。活字は前回と exact。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 認識の既定を `stage_recognize.py` に戻さない。
5. 認識既定が yomitoku を import しないことを戻さない。
6. 実体を git に載せない。

---

## 今動いている面

- 検出既定: 別置き ONNX（まだ yomitoku の DBnetPostProcessor）
- 認識既定: `stage_recognize_onnx.py` + `rec/`（yomitoku なし、torch 重みなし）
- 剥いだ後の活字は前回 lite と 37/37 exact。本文 3 つ残る
- 字表は `rec/resource/` に凍結（CC BY-NC-SA のコピー。latest を引かない）

---

## 次

検出後処理（DBnetPostProcessor）を剥ぐ。
検出既定が yomitoku を離れると、既定経路の役は cv2 / numpy / onnxruntime だけになる。
