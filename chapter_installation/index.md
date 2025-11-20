# 安装
:label:`chap_installation`

为了开始使用，我们需要搭建一个环境，用于运行 Python、Jupyter Notebook、相关*库* （library）以及运行本书本身所需的代码。

## 安装 Miniconda

最简单的选择是安装 [Miniconda](https://conda.io/en/latest/miniconda.html)。请注意，我们需要 Python 3.x 版本。 如果你已经在机器上安装了 conda，可以跳过以下步骤。

访问 Miniconda 网站，根据你的 Python 3.x 版本和机器*架构*（architecture）确定适合的版本。 假设你的 Python 版本是 3.9（我们测试所用的版本）。

如果你使用的是 macOS，你需要下载文件名包含字符串 "MacOSX" 的 bash 脚本，导航至下载位置，并按如下方式执行安装（以 Intel Mac 为例）：

```bash
# 文件名可能会有所变动
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Linux 用户需要下载文件名包含字符串 "Linux" 的文件，并在下载位置执行以下命令：

```bash
# 文件名可能会有所变动
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Windows 用户应按照其 [在线说明](https://conda.io/en/latest/miniconda.html) 下载并安装 Miniconda。在 Windows 上，你可以搜索 `cmd` 来打开命令提示符（命令行解释器）以运行命令。

接下来，初始化 Shell，以便我们可以直接运行 `conda` 。

```bash
~/miniconda3/bin/conda init
```


然后关闭并重新打开当前的 Shell。 你应该能够通过以下方式创建一个新环境：

```bash
conda create --name d2l python=3.9 -y
```


现在我们可以激活 `d2l` 环境：

```bash
conda activate d2l
```


## 安装深度学习框架和 `d2l` 包

在安装任何深度学习框架之前，请先检查你的机器上是否有合适的 GPU（标准笔记本电脑上用于驱动显示器的 GPU 通常不符合我们的需求）。例如，如果你的计算机拥有 NVIDIA GPU 并且已安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，那就一切准备就绪了。如果你的机器没有配备任何 GPU，也不用担心。 你的 CPU 提供了足够的算力来帮你完成前几章的学习。只要记住，在运行更大规模的模型之前，你将会需要使用 GPU。


:begin_tab:`mxnet`

要安装支持 GPU 的 MXNet 版本，我们需要知道你安装了哪个版本的 CUDA。 你可以通过运行 `nvcc --version` 或 `cat /usr/local/cuda/version.txt` 来检查。假设你安装的是 CUDA 11.2，那么执行以下命令：

```bash
# For macOS and Linux users
pip install mxnet-cu112==1.9.1

# For Windows users
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


你可以根据你的 CUDA 版本更改最后几位数字，例如，CUDA 10.1 对应  `cu101` ，CUDA 9.0 对应 `cu90` 。


如果你的机器没有 NVIDIA GPU 或 CUDA，你可以按如下方式安装 CPU 版本：

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

你可以按如下方式安装支持 CPU 或 GPU 的 PyTorch（指定的版本为编写时的测试版本）：

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:

:begin_tab:`tensorflow`
你可以按如下方式安装支持 CPU 或 GPU 的 TensorFlow：

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:

:begin_tab:`jax`
你可以按如下方式安装支持 CPU 或 GPU 的 JAX 和 Flax：

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


如果你的机器没有 NVIDIA GPU 或 CUDA，你可以按如下方式安装 CPU 版本：

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


我们的下一步是安装我们需要开发的 `d2l` 包，它封装（encapsulate）了本书中经常使用的函数和类：

```bash
pip install d2l==1.0.3
```

注意：这里的d2l要求numpy版本为1.23.5，可能会与更新版本的 Python 或 PyTorch 等库产生冲突。

## 下载并运行代码

接下来，你需要下载 Notebooks（笔记本），以便运行本书中的每一个代码块。只需点击 [D2L.ai 网站](https://d2l.ai/) 上任何 HTML 页面顶部的 "Notebooks" 标签即可下载代码，然后解压。 或者，你也可以按如下方式从命令行获取 Notebooks：

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

如果你还没有安装 `unzip` ，请先运行 `sudo apt-get install unzip` 。现在我们可以通过运行以下命令启动 Jupyter Notebook 服务器：

```bash
jupyter notebook
```


此时，你可以在 Web 浏览器中打开 http://localhost:8888（它可能已经自动打开了）。 然后我们就可以运行本书每一节的代码了。 每当你打开一个新的命令行窗口时，在运行 D2L notebooks 或更新你的软件包（无论是深度学习框架还是 `d2l` 包）之前，都需要执行 `conda activate d2l` 来激活运行时环境。 要退出环境，请运行 `conda deactivate`。


:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17964)
:end_tab:
