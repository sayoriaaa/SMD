## 下载

```
git submodule update --init --recursive
```

## 编译

windows： 双击运行`bash/build.bat` (请确保安装[mingw64](https://sourceforge.net/projects/mingw-w64/files/)和cmake并添加至环境路径)

若安装了CUDA toolkit（[我的使用版本](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)），则会自动识别并编译GPU程序。在windows下，需要安装VS Studio并将cl.exe添加至环境路径，并且先以管理员权限运行`cuda.bat`。

```
Hint: For new Visual Studio cl.exe is present in path => C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.23.28105\bin\Hostx64\x64
x64 is for 64bit
x86 is for 32bit
```

## 创建python环境 (可选)

我们使用python脚本来实现数据统计、可视化等任务

```
conda create -n SMD python=3.8
conda activate SMD
conda install -c conda-forge openmesh-python
pip install pyvista
```