## type
agent类型，若type取值为'xxx'，则对应agents/xxx.py

## update_mode
* 'unit'
    * 'episodes': 回合更新，如policy gradient算法
    * 'timesteps'
    * 'sequences'
* 'batch_size'
    * 当'unit' == 'episodes'时
        * 'batch_size'表示episodes number，即每次更新时，从memory里取最近的batch_size个episodes的数据
* 'frequency'
    * 若agent经历的episode数目为episodes_num，则满足episodes_num % frequency == 0的条件时更新参数
    * 参考tf_observe_timestep() in models/memory_model.py

## memory
* capacity: memory容量
* type
    * 'latest'
        * 存储：按照时间先后依次存入memory，若达到capacity值，则从头开始覆盖;　参考models/memory_model.py
        * 读取：self.memory_index points to the next record to be overwritten, 从self.memory_index记录的位置顺序读取
        * 参考: models/memory_model.py中tf_observe_timestep()函数
    * 'replay'
* include_next_states

## gae_lambda
表示使用Td error来估计Advantage Function, 参考tf_reward_estimation() in models/pg_model.py

## entropy_regularization
正则项系数
