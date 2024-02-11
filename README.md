# SMD-lab

<div align="center">

简体中文 | [English](README_en.md)
</div>







## 🌈简介

SMD-lab是一个基于libigl的毕业设计，旨在提供一个跨平台、开箱即用的保特征网格降噪算法工具，用于测试和比较不同算法在处理网格数据时的效果。

| 稀疏正则化                                                   | 压缩感知                                                     | 低秩分解                                                     | 滤波                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [(SIGGRAPH'13) Mesh denoising via *L*0 minimization](https://dl.acm.org/doi/10.1145/2461912.2461965) | [(SIGGRAPH'14) Decoupling Noises and Features via Weighted *l*1-analysis Compressed Sensing](http://staff.ustc.edu.cn/~lgliu/Projects/2014_DecouplingNoise/default.htm) | [(Proc. PG'18) Non-Local Low-Rank Normal Filtering for Mesh Denoising](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13556) | [(SIGGRAPH'03) Bilateral mesh denoising](https://dl.acm.org/doi/10.1145/882262.882368) |
| [(CAD'13) Feature-preserving filtering with L0 gradient minimization](https://dl.acm.org/doi/10.1016/j.cag.2013.10.025) |                                                              |                                                              | [(TVCG'11) Bilateral Normal Filtering for Mesh Denoising](https://dl.acm.org/doi/10.1109/TVCG.2010.264) |
|                                                              |                                                              |                                                              | [(Proc. PG'15) Guided Mesh Normal Filtering](http://staff.ustc.edu.cn/~juyong/GuidedFilter.html) |

该项目正在持续编写中，已复现的算法有：

- L0
- BF
- BNF
- GNF
- L0CDF（CAD'13）

## 使用

```
git submodule update --init --recursive
```

### 编译

windows： 双击运行`bash/build.bat` (请确保安装[mingw64](https://sourceforge.net/projects/mingw-w64/files/)和cmake并添加至环境路径)

若安装了CUDA toolkit（[我的使用版本](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)），则会自动识别并编译GPU程序。在windows下，需要安装VS Studio并将cl.exe添加至环境路径，并且先以管理员权限运行`cuda.bat`。

```
Hint: For new Visual Studio cl.exe is present in path => C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\bin\Hostx64\x64
x64 is for 64bit
x86 is for 32bit
```

### 创建python环境 (可选)

我们使用python脚本来实现数据统计、可视化等任务

```
conda create -n SMD python=3.8
conda activate SMD
conda install -c conda-forge openmesh-python
pip install pyvista
```

## 项目文件说明

- `src/` 源文件
  - `[论文算法]/`
  - `dependencies/` 
  - `utils/` 
- `data/` 数据集
- `bash/`
  - `*.bat` win下的任务
  - `*.sh` linux下的任务
- `scripts/` 脚本
- `run/`
  - `[具体任务]/`
    - `gt`
    - `noise`
    - `denoised`
    - `time.json` 记录每个网格去噪的时间开销



## 运行

### hello world!

在windows下，双击`bash/test_cube.bat` 

完成！ 你可以在当前命令行窗口看到这些算法的执行时间，以及对应降噪结果的不同指标（详细内容参见[降噪结果评估](#降噪结果评估)）你可以在`run/test_cube`下看到不同算法和不同参数的降噪结果。其中log文件夹下包含了HQS迭代求解时中间过程的所有能量和对应的网格模型。

接下来，通过执行`scripts/cube_vis.py`，可以生成以下的可视化结果:

各种算法对含$\sigma=0.7l_e$高斯噪声的网格的降噪结果，可视化为法向差异（$\degree$为单位）

![](imgs/gallery.png)

L0算法在迭代过程中的变化

![](imgs/L0-area-iter.png)

### 一般使用

SMD使用[clipp](https://github.com/muellan/clipp)来解析命令行参数，`build/`下包含所有的可执行文件，只需要在命令行下输入可执行文件名称，即可获得该程序的使用信息。



## 数据集

SMD支持两篇论文所提供的数据集：CNR提供的合成数据集Synthetic，扫描数据集Kinect v1、Kinect v2、Kinect F；GCN提供的扫描数据集PrintData。它们可以分别在https://wang-ps.github.io/denoising.html和https://drive.google.com/file/d/1x561-v3z1j0q_1qHYG0Fja1W-sqjhYpC/view下载

解压后，`data`下的文件为

```
└── data
    ├── examples
    ├── Kinect_Fusion
    ├── Kinect_v1
    ├── Kinect_v2
    ├── Synthetic
    └── PrintedDataset
```

PrintData数据集提供的噪声网格存在拓扑问题，对于一些算法会失效，因此需要先运行

```
python scripts/refine_pd.py
```

得到修复后的`PrintedDataset_r`数据集

通过`scripts/dataset.py`可一键运行数据集任务，使用`-h`查看帮助信息。e.g. 执行

```
python scripts/dataset.py --dataset PrintedDataset --metrics_args "--ahd --oep"
```

PS: 这里放弃了Thingi10K数据集，因为一方面已经有了Syhthetic数据集，一方面Thingi10K的数据集数量实在太多，全部跑完不现实，最后Thingi10K数据集中的网格不仅存在拓扑问题且网格质量不高（由样条转化而来）

PPS: PrintedDataset暂时不支持aad（尽管GCN中是有的，但不理解在F不一致下如何实现），期望作者能回邮件捏

## 降噪结果评估

### 平均豪斯多夫距离（AHD）

$$
E_v=\frac{1}{N_vL_d}\sum_{v^r_i\in v^r_M}
\min_{\tilde{v}_j\in\tilde{V}_m}\Vert v^r_i-\tilde{v}_j \Vert
$$

### 平均法向角距离（AAD）

$$
E_a=\frac{1}{N_f}\sum_{f^r_i\in F^r} \mathrm{acos}(n^r_i \cdot \tilde{n}_i)
$$

结果以角度为单位

### 翻折边比例（OEP）

基于$L_0$论文中folded triangle的可视化，使用边所对应的二面角进行网格评估，给出一个定量度量

$$
E_f=\frac{1}{N_e}\sum_{e^r_i\in F^r} \tau(e^r_i)
\\
\tau(e)=\begin{cases}
1&\mathrm{dihedral\ angle}(e)<30^\circ
\\
0&\mathrm{otherwise}
\end{cases}
$$



## TODO

BNF、GNF的全局方法实现



