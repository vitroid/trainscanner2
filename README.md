# trainscanner2

新しい TrainScanner のプロトタイプです。列車スキャン画像をより気軽に作成できるようになりました。

## インストール

```shell
pip install git+https://github.com/vitroid/trainscanner2.git
```

## 使い方

```shell
trainscanner2
```

実行すると、Drag&Drop 用のウィンドウが表示されますので、そこに動画ファイルを投げこんで下さい。

## 旧 TrainScanner に対する長所

- 全自動で、列車の検出と長画像化を行います。
- 動画の中に複数の列車があれば、個別に画像を生成します。
- 手ぶれはあらかじめ自動的に補正されます。
- より Robust なアルゴリズムにより、列車を見失って末尾が切れたりすることがかなり減りました。
- 傾きも自動的に補正されます。
- 高解像度画像も生成できます。
- できるだけメモリーを浪費しないように設計されています。
- YouTube の URL をドロップ(あるいはペースト)してスキャンできるようになりました。
  - 例: https://youtu.be/ME415Q1jCA4
  - https://www.youtube.com/shorts/uWddMNqK-8M
  - https://youtube.com/shorts/NcvS_FhGbVs?feature=share
  - https://youtube.com/shorts/wfGNFcdV8Y8?feature=share
  - https://youtube.com/shorts/4QFLOVuFhZM?feature=share
  - ひのとりミュージアムさんの完璧なサイドビュー https://www.youtube.com/shorts/oBDDACmtdjc

## 短所

順次改良します。

- スリット位置は指定できません。
- スリットのぼかし幅も指定できません。
- 大きな手ぶれや、とても速い列車、暗所での撮影には追随できません。
- 視野の中の小さな動体は見逃す可能性があります。
- それなりに処理が重いです。
- パース補正がありません。真横から撮影して下さい。
- 視野いっぱいに列車が写っていて、背景が分離できない場合にはうまく動きません(たぶん)
- 解像度を落してからつなぐので、列車移動速度が小さい場合には精度が落ちます。
- フレーム落ちに対応していません。
