# MADQN_for_Global_Routing
## 概要
## 実験条件の設定
### グリッドグラフ
問題設定の方法について記す．問題の作成に関する記述は全てmain.pyファイル内で行う．  
グリッドの大きさ，ピンペアの数，キャパシティの設定は以下で行う．
 - グリッドの大きさ＝35行目
 - ピンペアの数＝36行目
 - キャパシティ＝37行目

作成される問題の種類(ピンの位置)はseed値によって決まる．例えば5種類の問題を一度に実験する場合，  
158行目の変数seed_numに5つのseed値を含んだリストを与える．（ex.[0, 10, 20, 30, 40]）  

実験結果の保存先のファイル名は32行目に記述する．  デフォルトでは，自動で実行開始の日時を名前とするファイルが作成される．
ただし，端末によってエラーが出る場合があり，任意のファイル名を記載することで回避できる．
- (例)　dir_name = "ファイル名"
### A*アルゴリズム
A*アルゴリズムの試行回数はmain.pyファイルの49行目で設定できる．  
trail = N(本論文では10000)の場合，出力される結果はＮ回のうち，最もオーバーフロー（OF）が少なく，全配線長（WL）が短い経路である．
### DQN
シングルエージェント（DQN）の設定方法について記す．DQNに関する記述は全てDQN_fixed_order.pyとDQN_random_order.pyファイル内で行う．
 - バッチサイズ＝463行目
 - エピソード数＝464行目
 - １ピンペアにおけるステップ数＝465行目

 - Burn-in size＝326行目
 - Reply memory size＝327行目
 - 割引率＝328行目
 - ε＝329行目
 - 学習率＝332行目
### MADQN
マルチエージェント（MADQN）の設定方法について記す．MADQNに関する記述は全てMADQN.pyファイル内で行う．
 - バッチサイズ＝615行目
 - エピソード数＝616行目
 - １エピソードにおけるステップ数＝617行目

 - Burn-in size＝402行目
 - Reply memory size＝403行目
 - 割引率＝404行目
 - ε＝405行目
 - 学習率＝408行目
### 報酬の設定
報酬の設定方法について記す． 報酬の設定はmain.pyファイル内で行う．
 - if s' is the target pin＝23行目
 - if s' is an OF grid＝24行目
 - otherwise＝25行目

報酬はリストで与える． 複数の報酬設定で比較をしたい場合，   
(例：３種の報酬設定)
 - reward = [100, 100, 100]
 - penalty = [-10, -1, -0.1]
 - sparce_reward = [-1, -0.1, -0.01]  

のように記載する．
### 実行方法
以上の条件を設定したうえで， コマンドラインよりmain.pyを実行することで実験が開始される．  
## 結果の出力

## パッケージ
- Python implementation: CPython
- Python version       : 3.7.12
- IPython version      : 5.5.0

- torch: 1.10.0+cu111
- numpy: 1.19.5
- matplotlib: 3.2.2
