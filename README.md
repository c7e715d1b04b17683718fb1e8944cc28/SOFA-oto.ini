# SOFA-oto.ini
[SOFA](https://github.com/qiuqiao/SOFA)を使用して、UTAUのoto.iniを生成する

## 前提要件
- C++ によるデスクトップ開発 (Visual Studio)
- CMake

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
    pip install -r requirements.txt
    ```
4. [PyTorchの公式サイト](https://pytorch.org/get-started/locally/)にて、セットアップをする
5. [日本語のSOFAモデル](https://github.com/colstone/SOFA_Models/releases/tag/JPN-V0.0.2b)をダウンロードし、解凍後中にある「japanese-v2.0-45000.ckpt」を「src/cktp」に配置する
6. main.pyを実行する

## 仕様
UTAUの音源フォルダを本プログラムへ処理をかけることで、pyopenjtalkを用いてg2pを行いphonemesのテキストファイルを生成します。

その後、SOFAによってphonemesからラベリングを行い、ラベルファイルを元に独自の計算式でoto.iniへ変換します。