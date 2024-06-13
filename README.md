# SOFA-oto.ini
[SOFA](https://github.com/qiuqiao/SOFA)を使用して、UTAUのoto.iniを生成する

## 使い方 (Windows)
1. このリポジトリをsubmoduleを含めcloneする
    ```sh
    git clone --recursive
    ```
2. 仮想環境を構築し、入る
    ```sh
    python -m venv .venv
    .venv/scripts/activate
    ```
3. 必要なモジュールをインストールする
    ```sh
    pip install -r src/SOFA/requirements.txt
    pip install -r src/requirements.txt
    ```
4. [PyTorchの公式サイト](https://pytorch.org/get-started/locally/)にて、セットアップをする
5. main.pyを実行する

## 仕様
UTAUの音源フォルダを本プログラムへ処理をかけることで、pyopenjtalkを用いてg2pを行いphonemesのテキストファイルを生成します。

その後、SOFAによってphonemesからラベリングを行います。

TODO: その後ラベリングされたファイルからVCVやCVVCのoto.iniを生成します。