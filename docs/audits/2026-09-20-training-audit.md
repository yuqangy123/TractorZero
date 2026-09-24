**拖拉机训练停滞检查：2026-09-20**

检查对象：当前工作区提交 `7d71b28` 的执行路径 `train.py → dmc.train → act → Env/GameEnv/tractorGame`，以及 `log` 中的 TensorBoard、37 次评估和策略分布文件。没有修改训练、模型、环境或评估实现；新增的脚本只执行本地诊断，不启动训练、不读取或覆盖模型权重。

结论：已经复现游戏环境、样本奖励配对、模型决策目标、观测和评估中的多处错误。最值得优先处理的是“大局结束不重置等级”“叫主/埋牌奖励错位”“叫主 argmax 维度错误”“评估未进入 eval 模式”。这些足以解释为什么继续累计样本并不能稳定改善对规则胜率。各因素在这次训练中的贡献比例，需要原训练机上的检查点做消融评估才能确认。

**1. 日志实际说明了什么**

按每次评估的局数加权，将最早五次与最后五次比较；这是趋势摘要，不是独立同分布试验的显著性检验。相同固定发牌可能跨检查点复用，不能把这些局数当成完全独立的新样本。

| 指标 | 前五次加权胜率 | 后五次加权胜率 | 变化 | 对应局数 |
|---|---:|---:|---:|---:|
| banker_vs_rule | 24.05% | 24.68% | +0.62 个百分点 | 740 / 778 |
| banker_vs_random | 30.45% | 42.11% | +11.66 个百分点 | 624 / 532 |
| idler_vs_rule | 64.30% | 70.25% | +5.96 个百分点 | 717 / 790 |
| idler_vs_random | 76.20% | 84.30% | +8.11 个百分点 | 647 / 669 |

评估从 1,671,000 到 61,116,000 总 frames，共 37 个检查点。最后策略快照已到 62,478,000 frames。四个出牌角色的 loss 各有 126 个采样点，最后角色内步数均为 15,120,000。没有发现庄家单独停止训练的日志证据。最初文件列表被截断产生的怀疑已排除。

`dmc.py:458` 每消费一个角色的 batch 才给总 `frames` 加 `T*B`，所以六千万不是每个模型训练六千万，也不是六千万局。默认 `T*B=600`，每个出牌角色约有 25,200 次更新达到上述日志点；叫主和埋牌样本也计入总 frames。

庄家动作 loss 从 0.2222 降到 0.05995，牌型 loss 从 0.2311 降到 0.06365，但下面的目标错误使“loss 降低”不能推出“胜率目标优化成功”。日志中也有数值尖峰，例如 `banker_up/loss_act_loss` 最大 504.32、`banker_op` 最大 83.57；记录到的标量没有 NaN/Inf，但日志不是逐 batch 记录，不能据此排除未记录的异常。

**2. 大局结束不重置，长期训练集中于 A：环境错误，最高优先级**

位置：[botzone.py:537](D:/work/tractors/rlcard/games/tractors/env/botzone.py:537)、[botzone.py:484](D:/work/tractors/rlcard/games/tractors/env/botzone.py:484)、[env_utils.py:74](D:/work/tractors/rlcard/games/tractors/env/env_utils.py:74)。

`finalend` 分支只是把 `first_round` 设为 `None`，将 stage 改为 `gameend` 后再次 `reset()`。下一层 reset 判断的是键是否存在，因此并不会进入设置 level=2 的首局分支。双方 `player_level`、当前 level、庄家及累计系列状态没有作为新大局重建。训练包装器会持续 reset 同一个环境，而评估每次建立新 Env 并在第一次 finalend 停止。

实际运行底层环境连续 300 局合法单牌对局：第 36 局首次 finalend，之后没有再出现打 2，最后 100 局全部打 A。中间共出现 156 次 finalend。这是机制复现，不是从服务器日志统计到的实际等级分布；当前日志没有记录等级分布，不能给服务器训练作精确占比估计。

影响：运行越久，训练牌局越集中于高等级；评估却覆盖 2→A，形成明显分布不一致。单纯增加步数不能弥补中低等级样本不再进入训练的问题。

建议：把“新小局”和“新大局”重置分开；新大局重建双方等级、庄家、发牌及系列状态。记录每个级别的采样数。也可以明确改成按级别采样的独立小局训练，但要统一评估口径。

**3. 叫主和埋牌的奖励错配到下一局：训练数据错误，最高优先级**

位置：[act.py:193](D:/work/tractors/rlcard/games/tractors/act.py:193)、[act.py:207](D:/work/tractors/rlcard/games/tractors/act.py:207)。

结束一局时，先根据 `size[p] - len(target_wp_buf[p])` 填奖励，再把本局 bid/cover 的观测加入 buffer 并递增 size。因此第一次结束时没有待配对样本，第二局结束时才给第一局观测填入第二局 reward。这个一局偏移会跨 unroll 保留。

在真实 `act()` 和真实环境上，仅将终局 bid/cover 回报替换成便于辨认的编号：第 1～4 局应为 `[0.1, 0.2, 0.3, 0.4]`，实际训练 batch 是 `[0.2, 0.3, 0.4, 0.5]`。同时确认 batch 的观测确实来自第 1～4 局。bid 和 cover 都有这个问题。

建议：先登记本局样本、再填同局回报；或使用显式 episode_id 绑定样本和回报。出牌角色是在每次行动时递增 size，因此不能把这个结论泛化成所有出牌样本也都错一局。

**4. 叫主贪心决策恒选“不叫”，而且训练只保留每局最后一次决定**

位置：[bid_model.py:42](D:/work/tractors/rlcard/games/tractors/model/bid_model.py:42)、[act.py:154](D:/work/tractors/rlcard/games/tractors/act.py:154)、[botzone.py:1409](D:/work/tractors/rlcard/games/tractors/env/botzone.py:1409)。

候选动作输出是 `[N,1]`，却执行 `argmax(output, dim=1)[0]`。对只有一个元素的那一维取 argmax 必然得到 0。实测给三个候选分数 `[0.1,0.9,0.2]`，代码仍选第 0 个。合法列表第 0 个永远是空字符串“不叫”。

评估不传探索参数，因此当前代码下神经网络叫主永远不叫；所有人都不叫后，环境默认本局打无主。训练只有 epsilon 随机探索时才可能叫主，造成训练和评估叫主行为不同。

另外 `bid_obs_z/bid_obs_x` 被每一次叫主阶段行动覆盖。一局有 100 个决策点，但每局仅保存最后一个；真实 actor 复现中五局共执行 500 次 bid 决策，只有五个样本登记。早先成功叫主/反主的动作没有进入训练样本。只修 argmax 无法修复这条学习链路。

建议：沿候选动作维度选最大值；保存所有需要训练的叫主决策，至少保留存在实际选择的决策。回报要按决策者所属阵营归因，不能假定每个叫主行动者都是最终庄家。

**5. 评估模型保留 train 模式，导致同一个动作分数随候选 batch 改变**

位置：[hrl_agent.py:7](D:/work/tractors/rlcard/games/tractors/agents/hrl_agent.py:7)、[dmc.py:309](D:/work/tractors/rlcard/games/tractors/dmc/dmc.py:309)、[BasicBlockM.py:33](D:/work/tractors/rlcard/games/tractors/model/BasicBlockM.py:33)。

训练 actor 明确执行了 `model.eval()`，但 HRLAgent 创建/加载评估模型后没有调用 eval。网络有大量 BatchNorm；`torch.no_grad()` 不会切换 BatchNorm 的训练模式。

用真实 BankerModel 的固定随机权重与相同动作输入复现：仅把同一个动作从单独推理改成和其他候选一起推理，train 模式三头输出最大差值为 0.03882686；eval 模式差值约 3.9e-7。这不是已训练模型的胜率消融，而是明确证明候选集合正在影响评估分数。HRLAgent 的默认 training 状态也实际检查为 True。

`Model.play()` 对候选动作每 256 个分块，这还会让块之间的 BatchNorm 统计口径不同。评估的网络行为与训练采样时的网络行为不一致，因此现有曲线不能当成修复后模型的准确强度。

建议：加载权重后明确 eval；用同一检查点、同一固定牌局重新评估。候选分块前后的动作值与选择应保持一致。

**6. 出牌价值目标不等于整局胜率，所谓 win_rate 也不是胜率**

位置：[act.py:185](D:/work/tractors/rlcard/games/tractors/act.py:185)、[act.py:204](D:/work/tractors/rlcard/games/tractors/act.py:204)、[dmc.py:198](D:/work/tractors/rlcard/games/tractors/dmc/dmc.py:198)、[models.py:109](D:/work/tractors/rlcard/games/tractors/model/models.py:109)。

`target_adp` 是刚结束一墩的即时得分；没有累积剩余各墩回报，没有 bootstrap。win/lose 两头学习的是按终局结果分组的当前一墩得分。推理用 `p*win+(1-p)*lose` 排序，而不是直接按终局回报选择动作。胜负头确实受终局回报监督，不能说“完全没有终局信号”；问题是最终用于选动作的量把它混成了即时得分。

例如本墩无分时，win/lose 两头都接近 0，即使两个动作的终局价值不同，混合值也可能都接近 0。保留主牌、建立长套、控制最后一墩等跨墩收益难以通过当前目标得到正确排序。

更具体地，`target_wp` 是终局级差除 3：±1/3、±2/3、±1。模型却把它当作 ±1 胜负编码，用 `(win_rate+1)/2` 换成概率。一个必定普通赢一级的状态，正确回归值是 1/3，换算成“胜率”却是 2/3，而非 1。条件头的 loss 又乘了级差绝对值，分母按样本数量计，进一步改变了两头估计的量。

当前三头没有输出范围约束，所谓概率还可能小于 0 或大于 1。不能直接把所有回归头加 tanh 作为解决方案：实测末墩加底分后 `target_adp=1.0625`，即时目标本来就可能超过 1。

建议：先明确优化“胜率”还是“级差”。最简单的可验证基线是候选动作 Q 直接回归同局终局 Monte Carlo 回报，再 argmax；如保留分解头，就统一二值胜负概率与整局条件回报的定义、范围和权重。

**7. 上层策略没有控制下层，CE 学习的是事后结果分类**

位置：[play_model.py:380](D:/work/tractors/rlcard/games/tractors/model/play_model.py:380)、[play_model.py:404](D:/work/tractors/rlcard/games/tractors/model/play_model.py:404)、[dmc.py:217](D:/work/tractors/rlcard/games/tractors/dmc/dmc.py:217)。

BankerModel、IdlerModel 的牌型头和动作头虽然接收 strategy 参数，但拼接输入时完全没有使用它。把策略从 g1 切为 g7，真实 BankerModel 的两种下层输出差值均严格为 0。当前“分层策略”对固定权重的最终动作没有控制作用。

上层交叉熵标签又来自这墩结束后的赢家、是否有分牌等；价值项 gather 的是事后标签下标，不是 actor 实际选中的策略下标。注释声称是按回报优化的 Q(s,g)，实际代码不是这个算法。上层 loss 还会经共享牌型编码器影响下层，因此不仅是无用计算，也会改变特征学习。

建议：先把策略头作为独立诊断任务或暂时移除，建立直接 Q 基线；若保留分层决策，应真实接入所执行策略，并按执行策略配对回报。分类监督与 Q 监督不应混用同一组 logits 而缺乏一致目标。

**8. 埋牌输出范围与负回报冲突**

位置：[cover_model.py:38](D:/work/tractors/rlcard/games/tractors/model/cover_model.py:38)、[dmc.py:99](D:/work/tractors/rlcard/games/tractors/dmc/dmc.py:99)。

learn_cover 不传 flags，CoverModel 会 sigmoid 后再返回 values，即回归预测始终在 [0,1]；监督却是 ±1。负目标 -1 永远不可达到，持续把被埋牌的 logit 推向负饱和。八个选中牌位损失除以 120，在负回报样本上的理论下限仍为 8/120≈0.06667。

建议在修复同局奖励绑定后，统一成 0/1 概率目标或使用匹配回报范围的价值输出。选择埋牌还要对非法牌位做显式屏蔽；不能依赖乘以 0 与合法牌分数恰好保持正值。

**9. 观测确实有隐藏信息泄漏，同时缺门信息错误**

位置：[game.py:178](D:/work/tractors/rlcard/games/tractors/env/game.py:178)、[game.py:279](D:/work/tractors/rlcard/games/tractors/env/game.py:279)、[game.py:291](D:/work/tractors/rlcard/games/tractors/env/game.py:291)。

`mask_cards[r]` 从全部 108 张牌中删除了角色 r 的真实手牌，而输入又包含其他三家的这些 mask。对 mask 取补集即可恢复各家的初始真实手牌。庄家首个出牌观测中，三个其他角色均可精确恢复，已经用真实环境验证。

`other_hand_cards` 直接合并其他三家真实剩余手牌，没有包含未知底牌。闲家把这个集合与自己手牌、已出牌相加，再用全牌集取补，也能精确恢复底牌；已复现。若目标是正常暗牌升级，这是不合法观测。它未必单独造成训练内胜率低，却会改变问题定义，并污染修复后的公平评估及实际使用能力。

缺门处理还存在以下确定问题：处理上一动作时，`rule` 已被更新为下一行动者，删除牌却落到 `mask_cards[rule]`。复现：banker_op 垫牌，修改的是 banker_up 的 mask。牌花色取 `card % 4` 忽略第二副牌的 +54 偏移；应先归一化到单副牌编号。`level` 是字符串，但构造候选牌时与整数 `c` 比较，不能正确排除级牌。并且标记 DISCARD 只说明跟门张数不足，不一定说明本次出牌前一张该花色都没有。

建议：从该玩家可见信息构造未知牌集合和公开缺门推断；不要用对手真实分牌初始化输入。建立“固定公开信息和本手牌、只重分配隐藏牌，观测保持相同”的不变性检查。

**10. 裁判与结算的边界问题已复现**

位置：[botzone.py:800](D:/work/tractors/rlcard/games/tractors/env/botzone.py:800)、[botzone.py:950](D:/work/tractors/rlcard/games/tractors/env/botzone.py:950)、[botzone.py:599](D:/work/tractors/rlcard/games/tractors/env/botzone.py:599)、[botzone.py:2180](D:/work/tractors/rlcard/games/tractors/env/botzone.py:2180)。

按代码其他分支要求的“有门必须跟”规则：红桃 33、44 首出，跟牌者持红桃 6 和黑桃 33、44，黑桃是主牌。合法动作生成器只给出黑桃主拖拉机，保留红桃 6 不出；校验器错误码为 0。生成器在缺少同花色完整拖拉机时直接返回主拖拉机；校验器在跟出牌型恰好相同时绕过了“残余同花必须跟”的校验。这会制造错误的抢牌权/扣底路径。

甩牌罚分也会丢失。构造庄家甩红桃 3、5 且被其他家更大同花压住的牌局：首出后 `get_score=20`，但 `game_score=0`；下一名玩家 step 开头把 get_score 清零；这一墩结束后罚分仍没有进入总分，结果为 0。actor 只在墩末读取奖励，也收不到这次罚分。训练因此不能按预期成本惩罚失败甩牌。

另有应明确规则后再改的底分倍数问题：普通单张/对子/拖拉机使用牌张数作为乘数，甩牌分支却采用不同的倍数规则。不同升级变体约定不完全相同，这项不作为已确定的规则错误。

**11. 规则评估还有场面缓存、基线和指标口径问题**

位置：[simulation.py:182](D:/work/tractors/rlcard/games/tractors/eval/simulation.py:182)、[simulation.py:191](D:/work/tractors/rlcard/games/tractors/eval/simulation.py:191)、[rule_agent.py:35](D:/work/tractors/rlcard/games/tractors/agents/rule_agent.py:35)。

roundend 后又调用一次 `env.step()` 进入新墩，但没有重新获取传给 RuleAgent 的 infosets。二十局随机合法动作复现中，378 次墩间切换有 296 次缓存座位不是新行动者。网络使用新的 obs，规则对手却可能按旧手牌、旧座位和旧墩判断。这个比例来自诊断策略，并非日志中实际对局的比例。

规则对手 `_score_points` 把点数索引 7 当作 10：实际将 8 算成 10 分，将真正的 10 算成 0 分。它的拖拉机判断也直接用未剔除级牌的点数序列，与引擎 `pointorder` 不完全一致。应先统一规则基线，修复后对手强度也会变化，不能假定胜率只会向上。

`banker_vs_rule` 测的是庄家、对家、叫主、埋牌这套组合；`idler_vs_rule` 的叫主/埋牌来自规则模型。而 bid 实际为所有座位的叫主过程提供决策，不只是庄家自己的叫主。庄家侧坏掉的叫主/埋牌链路是两侧提升不对称的重要候选解释。应增加固定主花色/规则叫主扣底的纯出牌评估，以及只替换 bid、只替换 cover 的消融。

小局 winrate 按最后奖励符号累计，抽样核对没有发现简单的庄闲颠倒；双方天然胜率也不保证各 50%。但 `series_banker_final_level` 将动态庄家角色的带符号回报累加再加 2，当作真实系列等级，实际会出现 -34 等值。角色会随换庄变，终局奖励的 ±1/2/3 也不完全等于升级步数，所以这个 series 指标不能解读为真实等级或整轮胜率。

**12. 模型不是太小，先不要增加容量**

真实 BankerModel 有 110,560,792 个参数；单个动作编码器就有 30,769,336 个参数。四个独立出牌角色网络约 4.42 亿参数，另有 bid/cover。大量参数来自二维小牌面展平到 4096 的全连接以及宽 MLP；历史牌作为卷积通道输入，注释中的 LSTM/Attention/Transformer 没有进入当前出牌前向路径。

这可以成为样本效率、吞吐和泛化的改进方向，但不是当前首先应解决的问题。先修复目标、样本和评估，再考虑共享角色编码器、缩小网络、明确牌序/历史编码并做消融。不能用本次检查声称某个更大网络一定有效。

**13. 次要工程问题和检查限制**

- `strategy_dist/*.npz` 每次保存覆盖原文件后清空内存列表；本次各文件仅 5～6 行，不是完整六千万步策略历史。
- CPU actor 分支的 device 为字符串 `cpu`，创建进程却只处理整数 0/1；当前 Windows 入口会没有 actor 供数。这不解释来自 Linux 主机的这份历史曲线。
- 本机 rlcard 环境 Python 3.10 无法解析 `simulation.py:221` 的嵌套同引号 f-string；日志能正常产生说明不能据此断言服务器也是这个错误，须核对服务器 Python 版本。
- 评估先 join 子进程再读 Queue，大结果可能在 feeder flush 时阻塞；并行评估切分数据时，同一批内使用了相同 `started` 索引。默认 concurrency=1 时后一问题不触发。这些是结构风险，未从本次胜率曲线认定已发生。
- 评估权重路径写死，文件缺失时 HRLAgent 只打印后返回，可能继续用随机初始化参数；没有本次缺文件的证据。应记录加载状态/权重哈希并失败即停。
- 当前目录没有训练检查点，也没有服务器 console 日志或保存的实际 flags，无法确定每个检查点与当前源码完全一致、量化已训练权重的 BN 消融、或复算服务器的真实级别采样比例。

**14. 验证与建议顺序**

已完成：完整日志聚合；真实网络的叫主选取、策略条件、BatchNorm 行为和埋牌目标验证；真实 actor 五局奖励配对；二十局、1591 个随机出牌动作的牌守恒/终局奖励符号检查；300 局底层合法单牌运行；裁判和罚分的确定反例。

二十局普通路径没有触发非法动作异常，牌守恒与终局奖励阵营符号断言通过，但这不是规则正确性的证明——上面的反例说明生成器和裁判可能一起放行错误动作。

运行现有 `tests.models.test_tractors_strategy` 与 `tests.games.test_tractors_strategy_labels`：共 18 项测试，记录 7 个 failure 和 5 个 error（包含 subtest）。部分 error 是旧测试 flags 缺新增权重字段，部分断言仍要求旧的策略 ID/只按执行策略训练语义；这些不能直接视为服务器运行时异常。它们明确说明当前实现、注释和测试已经失配，尚无可据以宣布训练链路正确的通过结果。

建议依次执行：

1. 保留旧检查点和原始曲线；修复评估 eval 模式、场面缓存、模型加载检查，用旧权重建立可比基线。
2. 修复大局 reset、同局奖励绑定、叫主所有有效决策采样与 argmax、埋牌目标范围、跟牌和罚分规则、暗牌观测边界。为这些行为建立直接断言，记录级牌分布及每角色样本数。
3. 建立直接终局 Q 回归基线，统一优化胜率或级差，暂时去掉失效的分层策略。改变目标/观测后旧权重只能作为对照或 warm start，不能把历史 frames 连续拼成同一个实验。
4. 固定级别、发牌、随机种子、叫主扣底条件，分别评估纯出牌、bid、cover；记录实际局数、等级分布和不确定性。小规模实验出现稳定改善后再增加训练预算。
5. 最后做网络大小、参数共享、对手池与探索率的消融。当前证据不足以承诺修复后某个具体胜率。

复现命令（在仓库根目录执行）：

```powershell
& 'D:\ProgramData\Anaconda3\envs\rlcard\python.exe' tools/audit_tractors_training.py
& 'D:\ProgramData\Anaconda3\envs\rlcard\python.exe' -m unittest tests.models.test_tractors_strategy tests.games.test_tractors_strategy_labels
```

脚本：[audit_tractors_training.py](D:/work/tractors/tools/audit_tractors_training.py)。结构化证据：[2026-09-20-evidence.json](D:/work/tractors/docs/audits/2026-09-20-evidence.json)。脚本对自身测试运行里的无种子 RNG 做局部替换，使诊断可复现；不更改项目随机数实现。
