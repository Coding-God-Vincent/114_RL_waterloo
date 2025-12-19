# Part1
## DQN 結果
![]('Part1/DQN_result.png')  
## DRQN 結果  
![]('Part1/DRQN_result.png')  
## Discussion
本作業將環境從標準 CartPole 中隱藏了兩個速度資料使其成為一個 POMDP (Partial Observable MDP)。  
DQN 只能透過當前竿子的角度以及車子的位置來做為動作的依據，他沒辦法判斷竿子目前是上升還是下降，他只能瞎猜，導致獎勵只有到 45 而已。  
DRQN 透過 LSTM 捕捉時間序列的 feature 並透過該 feature 進行動作選擇的依據，因為該 feature 有歷史資料的資訊，這讓模型可以知道現在竿子是上升還是下降，並依照此資訊做出正確的判斷。可以由結果看出模型很好的學到正確的策略 (遊戲環境設定分數最高到 200)。  

---
# Part2
## DQN 結果
![]('Part2/DQN_result.png')  
## C51 結果
![]('Part2/C51_result.png')  
## Discussion
本作業將環境從標準的 CartPole 中加入了雜訊(摩擦力與隨機的托拉力量)，把原先的 deterministic env 變成了 stochastic env。C51 所學習的是 Q 值的機率分布並搭配 Cross-Entropy。相較之下，DQN 學習 Q 值得期望值，其所用的 MSE 會對離群值非常敏感。C51 的學習方法會讓更新的訊號相較 DQN 穩定，使得最終結果 C51 比 DQN 有更好更穩定的表現結果。