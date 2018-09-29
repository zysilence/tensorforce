## Reward
* 定义非常重要
* 计算hold_before时，是否考虑费用？费用高于价格增长的话，会倾向于不开仓
* TODO: 当“收益风险比”乘以“失败率” 的值“大于1”、“大于2”、“大于3”等等时，分别给予额外的奖励
* 要将交易费用算到reward中

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
* TODO: 尝试将蜡烛图变为2维输入，只有一个feature-map: (time_step, HLOC, 1)

## Episode
* Episode结束有几个条件：
    1. Finish one trading, e.g., bug once and sell once
    2. Reach the stop-loss line
    3. Reach the maximum episode length
    4. When testing, all testing data has been consumed
* 其中第1点是每个回合只进行一次交易，这是一种建模方式，符合人工交易时的思路。
    * 但是实验效果不好，现象是，基本上episode的长度为2左右，一买一卖结束
        * 原因可能是环境状态中没有返回当前交易的状态，即当前处于“多仓”状态，还是“空仓”状态，
agent没法根据这个信息来区分各个动作的含义。
    * 可尝试方法1：将交易状态作为环境状态的一部分这需要修改tensorce中的cnn网络：
        * 方法一：将交易状态作为cnn网络的另一维特征
        * 方法二：讲价格状态作为cnn网络的输入，在cnn输出端与交易状态拼接，然后输入到dense层和soft-max层
    * 可尝试方法2: 当episode没有结束时，每一步的reward设置为一个合适的正数，而不是0，从而激励agent尽量多持仓，而不是很快结束头寸
    * 可尝试方法3: 放弃该条件，每个回合可以交易多次
    * 以上几种方法根据效果来决定使用哪个
    
## Parameters Tuning
* Episode "max_len"
* Episode "trade_once"
* Episode "stop_loss_fraction"  
* Extra reward: when episode is not terminated, the agent gets extra reward
