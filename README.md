# 2D无粘不可压缩流体仿真

本项目为上海交通大学船舶海洋与建筑工程学院《高等计算流体力学》课程的探索性作业。  
项目使用Python与Taichi-lang实现了一个2D无粘不可压缩流的仿真。

本项目参考以下内容：
- GAMES201：高级物理引擎实战指南2020
- ETH Zurich：*Physically-Based Simulation in Computer Graphics HS21*

代码学习自项目[SandyFluid](https://github.com/ethz-pbs21/SandyFluid)，将其3D仿真重构为2D仿真，去除了PIC/FLIP算法，去除了沙模拟，去除了使用`connected-components-3d`等部分，使用`Taichi`模块的`GUI`重写了渲染与展示部分，使其成为一个最简化的2D流体仿真代码，可供学习使用。

## 安装

建议使用`conda`环境，并在环境中安装`Taichi`，可以参考[Taichi官方文档](https://docs.taichi-lang.org/)。

要求为：
- Python: 3.7/3.8/3.9/3.10 (64-bit)
- OS: Windows, OS X, and Linux (64-bit)

在`conda`中新建对应环境，并且切换至新环境中，安装`Taichi`：

```
pip install taichi
```

## 使用
下载本项目文件，进入项目根目录（即包含`src\`、`README.md`的目录）。

然后在安装了`Taichi`的命令行环境中执行命令：

```
python .\src\main.py
```

此命令将弹出一个窗口展示仿真过程。
