# Tractor 的上层策略训练

每次出牌按 `状态 s → strategy g → 牌型 tp → 具体动作 act` 决策。
`strategy_net` 输出七个策略的价值 `Q(s, g)`，不再输出专家标签的分类分数。
actor 用 epsilon-greedy 选定一个 `g`，同一个 one-hot 条件传给所有牌型候选和动作候选。
这是每次出牌重新选择的离散上层决策，不是跨多墩持续的 option。

## 采样与训练

- actor 保存出牌前的状态 `obs_strategy_z/x`，以及实际选中的整数 `strategy_action`，包括探索选择。
- 一局结束后，以当前角色的级数收益除以 3 得到 `target_wp`，填到该局每条出牌样本。
- 上层损失为 `mean((Q(s, strategy_action) - target_wp) ** 2)`。这是无折扣、终局收益的 DMC 回归；不是把当墩分数当长期回报，也不对未执行策略伪造回报。
- tp/act 使用采样时的 `strategy_action` 构造 one-hot，不使用 learner 当前重新预测的策略。下层原有的当墩得分/终局收益损失保持不变。
- 总损失为 `loss_tp + loss_act + loss_strategy`。离散选择不需要可微；上层通过自己的回报损失获得梯度。

`compute_strategy_targets` 的七种专家标签使用了出牌后的实际动作和赢家，只保留作行为诊断，不参与上述损失。
因此七个学习到的策略 ID 不保证继续对应“强势进攻”等固定名称；`strategy_sel` 是实际采样策略分布，`strategy_tgt` 是事后标签分布，两者不是分类准确率。
如果需要严格保持七种战术含义，还需要为下层定义各战术的动作约束或内在奖励，不能靠上层事后分类保证。

诊断标签中，“保底”指终局墩由我方（自己或队友）赢得；“垫牌”指非首出且自己未赢墩，包含跟门和未跟门的牌。
标签仍按既定优先级互斥：保底优先于垫牌，垫牌优先于同伴助攻；因此普通墩中非首出、自己未赢但队友赢墩时标为垫牌。

## 兼容性与验证

网络参数名和形状未变，旧权重可以加载，但旧策略头学到的是分类分数，尚未校准为回报；建议使用新实验目录训练并重新评估，不能直接认定旧 checkpoint 已适配新目标。
采样队列新增 `strategy_action`，升级时必须同时重启 actor 和 learner，不可混用旧队列。
异步 DMC 的目标来自采样时的上下层行为策略，并非无偏的当前贪心策略评估；需通过固定牌局评估判断收益是否提高。

回归测试：

```sh
python -m unittest discover -s tests/games -p test_tractors_strategy_labels.py -v
python -m unittest discover -s tests/models -p test_tractors_strategy.py -v
```

测试覆盖终局换庄/finalend、两副牌跟门、成功毙牌、真实行为策略对齐、上层梯度与回报方向、实际模型前后向，以及真实环境的跨局采样和 unroll 切片。
