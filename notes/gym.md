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

## 自定义环境
* 虚拟环境下gym的安装路径是: "$VIRTUALENV_PATH/lib/python3.6/site-packages/gym/envs/user/", 为了能够在工程中管理自定义代码，
需要将该目录建立一个软链接，链接到工程中的代码目录：
```shell
ln -s $PROJECT_PATH/gym_envs/user $VIRTUALENV_PATH/lib/python3.6/site-packages/gym/envs/user
```
* 由于修改了tensorforce的源码，同样为了工程中托管修改后的代码，需要建立该代码的软链接：
```shell
ln -s $PROJECT_PATH/tensorforce/execution/runner.py $VIRTUALENV_PATH/lib/python3.6/site-packages/tensorforce/tensorforce/execution/runner.py
```

