# conda 環境のセットアップ

### ターミナルをオープンして次のコマンドを次々実行します


conda init bash


conda create -n text_env python=3.8.*


conda activate text_env


### 次のコマンドは requirements.txt があるフォルダで実行します


pip install -r requirements.txt


### Jupyter カーネルを登録します


ipython kernel install --user --name text_env --display-name "Python (text_env)"


### 次のコマンドを実行します


sudo apt-get update


sudo apt-get install libmecab2 libmecab-dev mecab mecab-ipadic mecab-ipadic-utf8 mecab-utils
