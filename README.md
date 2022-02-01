# MADQN_for_Global_Routing
## 実験内容
*マルチエージェント（MADQN）によるグローバルルーティングの有効性を示すため，A*アルゴリズムとシングルエージェント（DQN）の配線結果を比較した．配線問題は，11×11のグリッド内で50ピンペアの接続を行う．キャパシティは5であり，1つのグリッドに合計5本までの配線が許容される．もし超えることがあれば，そのグリッド内で超過した回数だけオーバーフロー(OF)としてカウントされる．以上の問題を40種類無作為に生成し，OFの数及び全配線長の長さで評価する．評価の基準として，(1) 配線の混雑度 (2) 全配線長（WL）の最小化に成功した手法を最適な解とし，2つの異なる手法が同じ配線密度の制約を満たす場合，全配線長が短い方が良い解となる．混雑度と全配線長が同じの場合は，どちらの手法も最適な解とする．MADQNとDQNでは，１つの問題での学習におけるエピソード回数をどちらも1万エピソードとし，A*アルゴリズムは全配線経路の探索を1万回行うものとする．各手法で探索を行い，MADQNとDQNの場合は，1万エピソード内で累積報酬が最大となった経路を解とする．また，A*アルゴリズムとDQNでは配線順序の決め打ちが必要であるため，＜1＞ 2ピンの距離が遠い順（descent：降順）＜2＞ 2ピンの距離が近い順（ascent：昇順）＜3＞ 各試行で無作為に選んだ順（random）の３パターンで探索を行う．MADQNは同時に全配線を行うので，順番を決める必要が無い．

## 実験条件の設定
### グリッドグラフの設定
![grid_graph](/../main/images/grid_graph.png)  

問題設定の方法について記す．問題の作成に関する設定は全てmain.pyファイル内で行う．  
グリッドの大きさ，ピンペアの数，キャパシティの設定は以下で行う．  

![grid_graph](/../main/images/Grid_info.png)  

 - グリッドの大きさ＝35行目
 - ピンペアの数＝36行目
 - キャパシティ＝37行目

作成される問題の種類(ピンの位置)はseed値によって決まる．例えば5種類の問題を一度に実験する場合，  
158行目の変数seed_numに5つのseed値を含んだリストを与える．（ex.[0, 10, 20, 30, 40]）  

![grid_graph](/../main/images/seed.png) 

実験結果の保存先のファイル名は，32行目で設定する．  

![grid_graph](/../main/images/File_name.png)  

デフォルトでは，自動で実行開始の日時を名前とするファイルが作成される．
ただし，端末によってエラーが出る場合があり，任意のファイル名を設定することでエラーを回避出来る．
- (例)　dir_name = "ファイル名"  

### A*アルゴリズムの設定
A*アルゴリズムの試行回数はmain.pyファイルの49行目で設定する．  

![grid_graph](/../main/images/Astar_trial.png)  

trail = N(本論文では10000)の場合，出力する結果はＮ回のうち，最もOFが少なく，WLが短い経路である．  

### DQNの設定
シングルエージェント（DQN）の設定方法について記す．  
DQNに関する設定は全てDQN_fixed_order.pyとDQN_random_order.pyファイル内で行う．  

![grid_graph](/../main/images/DQN_epi_batch_set.png)  

 - バッチサイズ＝463行目
 - エピソード数＝464行目
 - １ピンペアにおけるステップ数＝465行目  
 
 ![grid_graph](/../main/images/DQN_set.png)  
 
 - Burn-in size＝326行目
 - Reply memory size＝327行目
 - 割引率＝328行目
 - ε＝329行目
 - 学習率＝332行目
### MADQNの設定
マルチエージェント（MADQN）の設定方法について記す．  
MADQNに関する設定は全てMADQN.pyファイル内で行う．  

![grid_graph](/../main/images/MADQN_epi_batch_set.png)  

 - バッチサイズ＝615行目
 - エピソード数＝616行目
 - １エピソードにおけるステップ数＝617行目  
 
 ![grid_graph](/../main/images/MADQN_set.png)  
 
 - Burn-in size＝402行目
 - Reply memory size＝403行目
 - 割引率＝404行目
 - ε＝405行目
 - 学習率＝408行目
### 報酬の設定
報酬の設定方法について記す． 報酬の設定はmain.pyファイル内で行う．  

![grid_graph](/../main/images/reward_func.png)  

![grid_graph](/../main/images/reward_set.png)  

 - if s' is the target pin＝23行目
 - if s' is an OF grid＝24行目
 - otherwise＝25行目

報酬はリストで与える． 複数の報酬設定で比較をしたい場合，   
(例：報酬を３種類設定する場合)
 - reward = [100, 100, 100]
 - penalty = [-10, -1, -0.1]
 - sparce_reward = [-1, -0.1, -0.01]  

のように設定する．  

### 実行方法
以上の条件を設定したうえで，コマンドプロンプトよりmain.pyを実行することで実験が開始される．  


## 結果の出力
### ファイルの構成
### task_info
ここでは，生成された問題のピンペアの位置を出力する．  
seed値を出力した後に各ピンペアの始点と終点の座標を出力する．  
（seed値が300，ピンペアの数が50の例）

![grid_graph](/../main/images/task_info.png)  

### congestion
ここでは，グリッド内における配線の混雑度を出力する.混雑度が高い程，色は黒に近づき，黒色（−１）のグリッドはOFを起こしてることを示す．  

![grid_graph](/../main/images/result_congestion.png)  

### route
ここでは，各ピンペアの配線経路を出力する．(以下に３つの例を示す．)  

![grid_graph](/../main/images/result_path1.png)  ![grid_graph](/../main/images/result_path2.png)  ![grid_graph](/../main/images/result_path3.png)  

### result
ここでは，配線結果におけるWL，OFの数，キャパシティ情報，接続出来たピンペアの数を出力する．  
例えば問題がピンペア数の50，キャパシティが5の場合，左から順に
 - seed値
 - WL
 - OFの数
 - ワイヤが５本引かれたグリッドの数
 - ワイヤが４本引かれたグリッドの数
 - ワイヤが3本引かれたグリッドの数
 - ワイヤが2本引かれたグリッドの数
 - ワイヤが1本引かれたグリッドの数
 - ワイヤが0本引かれたグリッドの数
 - 接続出来たピンペアの数  
 
 ![grid_graph](/../main/images/result_result.png)  
 
### log(DQN&MADQN)
ここでは，学習におけるseed値，エピソード数，エピソード毎の累積報酬，WL，接続出来たピンペアの数を出力する．  
seed値を出力した後に左から順に
 - エピソード数
 - 累積報酬
 - WL
 - 接続出来たピンペアの数  
 
 ![grid_graph](/../main/images/result_log.png)  
 
### reward(DQN&MADQN)
ここでは，seed値とエピソード毎の累積報酬を出力する．
## パッケージ
- Python implementation: CPython
- Python version       : 3.7.12
- IPython version      : 5.5.0

- torch: 1.10.0+cu111
- numpy: 1.19.5
- matplotlib: 3.2.2
