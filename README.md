# selecting_flow_example
## How to Run
1. Rustのインストール [rustup](https://rustup.rs/) からやるのが楽なはず
1. データセットのダウンロード 動作検証は [kaggle](https://www.kaggle.com/c/extreme-classification-amazon) よりダウンロードしたものでやっています 
```bash
$ git clone https://github.com/White-Green/selecting_flow_example
$ cd selecting_flow_example
$ cargo run --release -- --feature path/to/trn_ft_mat.txt --label path/to/trn_lbl_mat.txt
```

# Pytorch版について
比較検証に使ったpytorchのコードがverify_pytorchディレクトリにあります。
作者の力量不足によりRust版ほど使いやすくなっていませんので動作させるためにいくつか書き換える必要があります。
- GPUで動かす場合、L.72,79のコメントアウトを外し、L.73をコメントアウトしてください
- データセットの場所を指定するために、L.123の文字列を変更してください
