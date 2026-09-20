# 次のセッションへ

2026-09-20: 既定の画素面を cvsurf に移した。pip の opencv-python は既定から外した。
公式 .so は Qt/ffmpeg がリンカで付いてくるので、間引き配送は捨てた。

---

## 絶対に戻さないこと

1. 検出と認識を同じプロセスに載せない。
2. AR 展開 ONNX を既定にしない。
3. 検出の既定を torch に戻さない。
4. 認識の既定を torch / yomitoku に戻さない。
5. 既定の検出・認識が yomitoku を import しないことを戻さない。
6. 結合後の実体を git に載せない。50MB 以上の単一ファイルを載せない。
7. 既定 verify に safetensors を強制しない。
8. comparison/ の外で yomitoku / torch を import しない。
9. 既定に pyclipper を戻さない。
10. 既定経路に opencv-python を戻さない。画素は cvsurf。
11. この部品に場面語を混ぜない。
12. README を作業台に戻さない。SESSION を消さない。

---

## 今動いている面

- origin clone → join → verify の重み路はそのまま
- 既定 import は `from cvsurf import cv2`
- runtime pip は numpy と onnxruntime だけ
- 公式 cv2 との画素 exact は保証しない。remain は新しい画素で取り直す

---

## 次

settei21 で remain を取り直す。JPEG 復号と輪郭が公式と違うので、箱数が動く可能性がある。
