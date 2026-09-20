# 別置き重み

git にはハッシュだけを置く。実体は `pinned/` である。

入手:

```
python3 weights/pin_weights.py
python3 weights/verify_weights.py
```

無い・壊れているときは失敗する。黙って最新を取りに行かない。
入口は検出の前に verify を呼ぶ。
