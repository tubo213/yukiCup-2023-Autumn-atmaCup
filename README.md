# yukiCup 2023 Autumn x atmaCup

## Build Environment
### 1. install [rye](https://github.com/mitsuhiko/rye)

[install documentation](https://rye-up.com/guide/installation/#installing-rye)

MacOS
```zsh
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.zshrc
source ~/.zshrc
```

Linux
```bash
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
source ~/.bashrc
```

Windows  
see [install documentation](https://rye-up.com/guide/installation/)

### 2. Create virtual environment

```bash
rye sync
```

### 3. Activate virtual environment

```bash
. .venv/bin/activate
```

### Set path
Rewrite run/conf/dir/local.yaml to match your environment

```yaml
data_dir: 
processed_dir: 
output_dir: 
```

## Prepare Data

### 1. Download data

./data/ 配下にデータを解凍してください

### 2. Preprocess data

```bash
rye run python run/prepare_data.py 
```

## Train

以下のコマンドでlocal score=0.41, lb=0.318のモデルが学習できます。
output/train/exp001に提出ファイルが生成されます。
```bash
rye run python run/train.py epochs=15
```

[hydra](https://hydra.cc/docs/intro/)を利用しているため、パラメータを変更した実験が簡単に行えます。
以下のコマンドは、lrを0.1, 0.01, 0.001で実験を行います。
```bash
rye run python run/train.py -m lr=0.1,0.01,0.001
```
