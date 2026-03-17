# lingk

`lingk` は、局所フラックスチューブモデルにおける線形ジャイロ運動論方程式ソルバです。
このリポジトリには、Fortran による本体実装と、gnuplot を用いた簡易可視化スクリプトが含まれています。

## リポジトリ構成

- `src/`: ソルバ本体の Fortran ソースコード
- `param.namelist`: 実行時に読み込む入力パラメータ
- `Makefile`: Intel Fortran (`ifx`) を既定としたビルドルール
- `plot_*.gn`: 出力確認用の gnuplot スクリプト

## ビルド

既定の `Makefile` では Intel Fortran を使います。

```bash
make lingk
```

これにより `lingk.exe` が生成されます。

GNU Fortran を使いたい場合は、[`Makefile`](/home/smaeyama/github/test_lingk/Makefile) 内の
`gfortran` 用設定を有効にし、`ifx` 用設定を無効にしてください。

## 実行

ソルバは [`param.namelist`](/home/smaeyama/github/test_lingk/param.namelist) から物理パラメータを読み込み、
出力を `./data/` 以下に書き出します。

```bash
mkdir -p data
./lingk.exe
```

付属のサンプル入力では、たとえば以下の値が設定されています。

- `ky = 0.2`
- `eps_r = 0.18`
- `q_0 = 1.4`
- `s_hat = 0.8`
- `R0_Ln = 2.2`
- `R0_Lt = 6.9`

## 出力ファイル

既定の `flag_runs = 1` では、Fortran ソルバは次のファイルを出力します。

- `data/frq.001`: 線形成長率と周波数の履歴
- `data/mominzt.001`: `z` と時刻に対する場と密度モーメント
- `data/fkinzv_imXXXX_tYYYYYYYY.dat`: 指定した `mu` インデックスにおける分布関数のバイナリ出力

主要な数値パラメータは
[`src/parameters.f90`](/home/smaeyama/github/test_lingk/src/parameters.f90)
で定義されています。たとえば以下のような値です。

- `nz = 24 * 5`
- `nv = 32`
- `nm = 31`
- `dt_out = 0.1`
- `time_limit = 10.0`

## 可視化

gnuplot が使える環境であれば、付属スクリプトで簡単に出力を確認できます。

```bash
gnuplot plot_linfreq.gn
gnuplot plot_mominz.gn
gnuplot plot_fkinzv.gn
```

これらのスクリプトは、既定の Fortran 出力が `./data/` にあることを前提としています。
