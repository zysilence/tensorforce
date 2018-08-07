## TimeLimit
* 每个env实例被一个名为TimeLimit的Wrapper封装, 可以限制每个episode中step的最大值和reward最大值
* 被wrap后的env执行step()方法时，会调用TimeLimit.step(),该函数会判断当前经历过的step数(self._elapsed_steps)
与最大step数(self._max_episode_steps)值，超过最大值则退出
* self._max_episode_steps的值在gym/envs/__init__.py中设置，默认值是200
* 若不想限制最大step的数量，则可以使用未经过wrap的env实例：
    ```python
    env = env.unwrapped
    ```
    'unwrapped'字段表示没有经过wrap的原始的env
