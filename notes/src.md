### Model
* 定义和训练模型
* Model.__init__ ----> Model.setup() ----> Model.setup_components_and_tf_funcs()
    * 初始化阶段定义了模型结构和模型训练相关的tensorflow operations

### multy_step.py
* tf_step()
    * 一个训练样本参数更新多次，类似于监督学习中的epochs，同一个训练样本利用多次。
      在蒙特卡洛方法中，往往没有epoch的概念，取而代之的是episode的概念，为了提高
      训练的效率，可以每个episode更新多次。
    * 在Policy Gradient with Baseline方法中，估计baseline(即V(s))时，使用了multi_step

### Vanilla Policy Gradient
* agent: vpg.agent
* Loss function
    * 采用average reward per time_step, 即 Sum(log(pi)*advantage_value(s))/len(batch)
* 不使用baseline时
    * 使用discouted cumulative reward作为Q(s,a)的采样
    * 一个batch为一个episode，每个或者几个(取决于配置)episode结束后更新一次参数
* 使用baseline时
    * baseline为V(s)
    * 使用MLP作为V(s)的估计
    * 每个episode结束，策略更新一次的同时， MLP的参数更新num_steps次
