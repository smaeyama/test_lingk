# dev_python_lingk

`dev_python_lingk/` は、`lingk` の Python 参照実装と、その出力確認・可視化用スクリプトをまとめた作業用ディレクトリです。
リポジトリ全体としては Fortran 版が主ですが、このディレクトリでは Python 版を独立に扱えるようにしています。

## ファイル構成

- `lingk.py`: Python 版ソルバ本体
- `check_fortran_vs_python.py`: Fortran 出力との差分確認
- `plot_mominz.py`: `mominzt.nc` の可視化
- `plot_fkinzv.py`: `fkinzv.nc` の可視化
- `plot_linfreq.py`: `frq.txt` のマルチプロット
- `test/test_fortran_python_equivalance.py`: Python 版と Fortran 出力の整合確認
- `requirements.txt`: Python 側で使う主な依存パッケージ

## 実行例

リポジトリルートで Python 版を実行する場合:

```bash
python3 dev_python_lingk/lingk.py --param-namelist param.namelist
```

`dev_python_lingk/` に移動して実行する場合:

```bash
cd dev_python_lingk
python3 lingk.py --param-namelist ../param.namelist
```

既定では `lingk_output/` 以下に次のファイルが出力されます。

- `lingk_output/frq.txt`
- `lingk_output/mominzt.nc`
- `lingk_output/fkinzv.nc`

## 可視化

`dev_python_lingk/` にいる場合は、たとえば以下のように実行できます。

```bash
python3 plot_linfreq.py
python3 plot_mominz.py
python3 plot_fkinzv.py
```

`plot_linfreq.py` は、成長率・周波数・収束指標を 1 枚のマルチプロットとして表示します。
各スクリプトとも、既定では画面表示を行い、`--save` を付けると画像やアニメーションを保存できます。
また `--no-show` を付けると画面表示を抑止できます。

例:

```bash
python3 plot_linfreq.py --save figures/frq.png --no-show
python3 plot_mominz.py --save movies/mominz.gif
python3 plot_fkinzv.py --save movies/fkinzv.gif
```

## Fortran 版との比較

リポジトリルートの `data/` に Fortran 出力がある前提で、次のように比較できます。

```bash
python3 dev_python_lingk/check_fortran_vs_python.py --assert-match
```

既定のしきい値は、Fortran 側テキスト出力の丸め誤差を考慮した設定になっています。

## テスト

`dev_python_lingk/` に移動して `pytest` を実行すると、`test/` 配下の比較テストが走ります。

```bash
cd dev_python_lingk
pytest
```

テスト中の Python 版実行は短時間で済むように、`--max-steps 100` を使う設定にしています。
