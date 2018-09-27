## Reward
* 定义非常重要
* 计算hold_before时，是否考虑费用？费用高于价格增长的话，会倾向于不开仓

## Action
* 对于PPO算法，训练时是随机，测试时是随机还是确定的？

## Debug information
* 打印动作信息
* 打印回合收益

## Position(仓位)
* 哪种策略更好？
    * 每次交易额为初始资金的固定比例
    * 每次交易额为当前资金的固定比例
    
## Data
* 非交易时间的数据怎么处理？
    * 与最近交易时间的数据对齐
    * 提出该数据 
* period可以配置：1分钟、5分钟、15分钟、30分钟、1小时、4小时、1天
* 不同周期的特征放一起？
* 使用不同大小的feature map来提取不同feature？

## Network
* 设计CNN网络结构
* 尝试将蜡烛图变为2维输入，只有一个feature-map: (time_step, HLOC, 1)

## Episode
* 尝试不设置长度的最大值？

## Parameters Tuning
