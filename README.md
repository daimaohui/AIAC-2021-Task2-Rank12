# AIAC-2021-Task2-Rank12

## 初赛技巧

​	1.model拟合过程使用**决策树**进行拟合。

​	2.好的优化器可以保持最优值很稳定在一个区间里面，推荐使用**skopt.gbrt_minimize**

​    3.保证模型能够拟合比较好，可以浪费一定的找最大值的次数，比如前10次全部为随机初始化过程。

​    4.线下和线上差异很大，对于**决策树**来说线下只能到**0.2**左右，线上在**0.6**左右浮动

| 模型         | 优化器                  | 线下 | 线上          |
| ------------ | ----------------------- | ---- | ------------- |
| gp           | scipy.optimize.minimize | 0.3  | 0.25          |
| DecisionTree | scipy.optimize.minimize | 0.2  | 0.51-0.68区间 |
| DecisionTree | skopt.gbrt_minimize     | 0.45 | 0.55-0.68区间 |

## 决赛技巧

​	主要借鉴代码：[北大openbox开源地址](https://github.com/PKU-DAIR/open-box)

​	1.由于次数的减少，导致**决策树**的拟合效果变差，尝试了许多模型，效果不是很明显的，线上大概在**0.31**左右

​	2.决赛没有什么高级的想法，主要是将openbox的代码看懂上层的一部分，然后进行一部分的改写，**只能说越看别人的代码，越感觉到自己的菜**。

​    3.改写主要是改写openbox中**optimizer.** **generic_smbo.py core.** **generic_advisor.py**这个两个文件

​       

| **optimizer.** **generic_smbo.py** | run_init()<br/>iterate_init()<br/>add_history()<br/>run()<br/>iterate() |
| ---------------------------------- | ------------------------------------------------------------ |
| **core.** **generic_advisor.py**   | **get_suggestion()**                                         |

​	4.使用openbox模型为**gp**,优化器为**batchmc**	

​	5.没有任何的**is_early_stop**的实现

| 模型         | 优化器               | 线下 | 线上             |
| ------------ | -------------------- | ---- | ---------------- |
| DecisionTree | skopt.gbrt_minimize  | 0.4  | 0.3-0.31区间     |
| openbox.gp   | openbox.local_random | 0.58 | 0.03170000(超时) |
| openbox.gp   | openbox.batchmc      | 0.9  | 0.53             |
|              |                      |      |                  |

