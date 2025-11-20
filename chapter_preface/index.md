# 前言

就在几年前，在大公司和初创企业中，还没有成群结队的深度学习（Deep Learning）科学家在开发智能产品和服务。当我们进入这一领域时，机器学习（Machine Learning）并未占据日报的头条版面。我们的父母完全不知道机器学习是什么，更不明白为什么我们宁愿选择它，而不去从事医学或法律等职业。那时的机器学习是一门“仰望星空”的纯学术学科，其在工业界的意义仅限于语音识别和计算机视觉等少数现实应用。此外，许多此类应用往往需要大量的领域知识，以至于它们常被视为完全独立的领域，而机器学习只是其中很小的一部分。在那个时候，神经网络（Neural Networks）——也就是本书所重点关注的深度学习方法的前身——普遍被认为是过时的技术。

然而，短短几年间，深度学习给世界带来了巨大的惊喜，推动了计算机视觉、自然语言处理、自动语音识别、强化学习和生物医学信息学等多个领域的快速发展。此外，深度学习在众多实际任务中的成功，甚至催生了理论机器学习和统计学的发展。凭借这些进步，我们现在可以制造比以往任何时候都更自动化的汽车（尽管其自动化程度可能不如某些公司宣称的那样高）、能通过提出澄清性问题来调试代码的对话系统，以及在围棋等棋类游戏中击败世界顶尖人类选手的软件智能体——这曾被认为还需要几十年才能实现。这些工具已经对工业界和社会产生了日益广泛的影响，改变了电影制作、疾病诊断的方式，并在天体物理学、气候建模、天气预报和生物医学等基础科学中发挥着越来越重要的作用。

## 关于本书

本书代表了我们试图让深度学习变得平易近人的尝试，我们将向你传授 *概念*（concepts）、*背景*（context） 和 *代码*（code）。

### 代码、数学和 HTML 三合一的媒介

任何一种计算技术要发挥其最大影响力，都必须被充分理解、有详尽的文档，并有成熟、维护良好的工具支持。关键思想应被清晰地提炼出来，最大限度地缩短新从业者跟上进度所需的上手时间。成熟的库应该能自动化处理常见任务，而示例代码应使从业者能够轻松修改、应用和扩展常见的应用程序以满足即时需求。

以动态 Web 应用程序为例。尽管亚马逊等许多公司在 20 世纪 90 年代就开发了成功的数据库驱动的 Web 应用程序，但这项技术帮助创造型企业家的潜力直到过去十年才得到更大程度的释放，这在很大程度上归功于强大且文档完善的框架的发展。

探索深度学习的潜力面临着独特的挑战，因为任何一个单一的应用都会汇集多个学科。应用深度学习需要同时理解： (i) 以特定方式通过模型解决问题的动机； (ii) 给定模型的数学形式； (iii) 将模型拟合到数据的优化算法； (iv) 统计学原理，告诉我们何时可以预期模型能泛化到未见过的数据，以及验证其确实具有泛化能力的实用方法； (v) 有效训练模型所需的工程技术，包括驾驭数值计算的陷阱以及最大限度地利用可用硬件。 要在同一个地方同时教授制定问题所需的批判性思维技能、解决问题的数学知识以及实现解决方案的软件工具，这是一项艰巨的挑战。本书的目标是提供一个统一的资源，帮助原本想要入门的从业者快速上手。

当我们开始这个写书项目时，还没有任何资源能够同时满足以下条件： (i) 保持内容不过时； (ii) 既覆盖现代机器学习实践的广度，又有足够的技术深度； (iii) 将教科书应有的高质量阐述与实操教程应有的整洁可运行代码相结合。 我们发现了很多代码示例，说明如何使用给定的深度学习框架（例如，如何在 TensorFlow 中使用矩阵进行基本数值计算）或实现特定技术（例如，LeNet、AlexNet、ResNet 等的代码片段），但它们散落在各种博客文章和 GitHub 仓库中。然而，这些示例通常侧重于 如何 实现给定方法，却忽略了关于 为什么 做出某些算法决策的讨论。虽然偶尔会出现一些针对特定主题的交互式资源（例如网站 [Distill](http://distill.pub), 上发布的引人入胜的博客文章或个人博客），但它们仅涵盖深度学习中的选定主题，且往往缺乏配套代码。另一方面，虽然出现了一些深度学习教科书——例如 :citet:`Goodfellow.Bengio.Courville.2016`, 它提供了关于深度学习基础知识的全面综述——但这些资源并未将描述与代码中的概念实现结合起来，有时会让读者对如何实现这些概念一头雾水。此外，太多的资源被隐藏在商业课程提供商的付费墙之后。

我们需要创建一个能够满足以下条件的资源： (i) 所有人均可免费获取； (ii) 提供足够的技术深度，作为真正成为应用机器学习科学家道路上的起点； (iii) 包含可运行的代码，向读者展示 如何 在实践中解决问题； (iv) 允许我们以及整个社区进行快速更新； (v) 配有一个[论坛](https://discuss.d2l.ai/c/5)，用于互动讨论技术细节和回答问题。

这些目标往往是相互冲突的。公式、定理和引用最好用 LaTeX 来管理和排版。代码最好用 Python 描述。而网页的原生语言是 HTML 和 JavaScript。此外，我们希望内容既可以作为可执行代码访问，也可以作为实体书、可下载的 PDF 以及互联网上的网站访问。似乎没有现成的工作流能满足这些需求，所以我们决定构建自己的工作流 (:numref:`sec_how_to_contribute`)。我们决定使用 GitHub 来共享源代码并促进社区贡献；使用 Jupyter notebooks 来混合代码、公式和文本；使用 Sphinx 作为渲染引擎；并使用 Discourse 作为讨论平台。虽然我们的系统并不完美，但这些选择在相互竞争的关注点之间取得了折衷。我们相信，《动手学深度学习》（Dive into Deep Learning）可能是第一本使用这种集成工作流出版的书籍。


### 干中学

许多教科书按顺序呈现概念，对每个概念都进行详尽的介绍。例如，:citet:`Bishop.2006` 的优秀教科书对每个主题的讲解都非常透彻，以至于要读到线性回归这一章需要完成大量不简单的前置工作。虽然专家们正是因为这种透彻性而喜爱这本书，但对于真正的初学者来说，这一特性限制了它作为入门教材的实用性。

在本书中，我们采用 *适时* （just in time）教学法。换句话说，你将在需要某个概念来实现某些实际目标的那一刻学习它。虽然我们在开始时会花一些时间讲授基本的预备知识，如线性代数和概率论，但我们希望你在担心更深奥的概念之前，先尝到训练出第一个模型的满足感。

除去几个提供基本数学背景速成课程的预备笔记本（notebook）外，随后的每一章都会介绍适量的新概念，并提供几个使用真实数据集的、自包含的工作示例。这带来了一个组织上的挑战。有些模型在逻辑上可能归为同一个笔记本。有些想法可能最好通过连续执行几个模型来教授。相比之下，坚持 *一个工作示例，一个笔记本* 的策略有很大的优势：这使你尽可能容易地利用我们的代码开始你自己的研究项目。只需复制一个笔记本并开始修改它即可。

在整本书中，我们将可运行的代码与所需的背景材料穿插在一起。通常，我们倾向于在完全解释工具之前先提供它们（稍后往往会补充背景知识）。例如，我们可能会先使用 *随机梯度下降*（stochastic gradient descent），然后再解释它为什么有用或提供一些关于它为何有效的直觉。这有助于为从业者提供快速解决问题所需的弹药，代价是要求读者先信任我们的一些策划决定。

本书从零开始教授深度学习概念。有时，我们会深入探讨那些通常被现代深度学习框架对用户隐藏的模型细节。这在基础教程中尤为常见，因为我们希望你理解给定层或优化器中发生的一切。在这些情况下，我们通常会提供示例的两个版本：一个是我们从头开始实现所有内容，仅依赖类似 NumPy 的功能和自动微分；另一个是更实用的示例，我们使用深度学习框架的高级 API 编写简洁的代码。在解释了某个组件的工作原理后，我们在随后的教程中将依赖高级 API。

### 内容与结构

本书大致可分为三个部分，分别涉及预备知识、深度学习技术以及侧重于真实系统和应用的高级主题 (:numref:`fig_book_org`).

![全书结构](../img/book-org.svg)
:label:`fig_book_org`


* **第一部分：基础与预备知识**.
:numref:`chap_introduction` 是深度学习的介绍。然后，在 :numref:`chap_preliminaries` 中，我们会让你快速掌握动手进行深度学习所需的先决条件，例如如何存储和操作数据，以及如何基于线性代数、微积分和概率论的基本概念应用各种数值运算。:numref:`chap_regression` 和 :numref:`chap_perceptrons`
涵盖了深度学习中最基本的概念和技术，包括回归和分类；线性模型；多层感知机；以及过拟合和正则化。

* **第二部分：现代深度学习技术。**.
:numref:`chap_computation` 描述了深度学习系统的关键计算组件，并为我们随后实现更复杂的模型奠定基础。接下来，:numref:`chap_cnn` 和 :numref:`chap_modern_cnn` 介绍了卷积神经网络（CNN），这是构成大多数现代计算机视觉系统骨干的强大工具。同样，:numref:`chap_rnn` and :numref:`chap_modern_rnn` 介绍了循环神经网络（RNN），这是一种利用数据中的序列（例如，时间）结构的模型，常用于自然语言处理和时间序列预测。在:numref:`chap_attention-and-transformers`, 我们描述了一类相对较新的模型，基于所谓的 注意力机制（attention mechanisms），它已取代 RNN 成为大多数自然语言处理任务的主流架构。这些章节将让你跟上深度学习从业者广泛使用的最强大、最通用的工具。

* **第三部分: 可扩展性、效率与应用** （[在线](https://d2l.ai) 可用）。在第 12 章中，我们将讨论用于训练深度学习模型的几种常见优化算法。接下来，在第 13 章中，我们将研究影响深度学习代码计算性能的几个关键因素。然后，在第 14 章中，我们将展示深度学习在计算机视觉中的主要应用。最后，在第 15 章和第 16 章中，我们将演示如何预训练语言表示模型并将其应用于自然语言处理任务。


### 代码
:label:`sec_code`

本书的大部分章节都配有可执行代码。我们相信，通过试错、微调代码并观察结果，最能培养直觉。理想情况下，优雅的数学理论可能会告诉我们究竟如何调整代码以获得预期的结果。然而，今天的深度学习从业者往往必须在缺乏坚实理论指导的情况下探索。尽管我们尽了最大努力，但通过数学来表征这些模型的难度很大，目前仍缺乏对各种技术有效性的正式解释；解释可能取决于目前缺乏清晰定义的数据属性；且关于这些主题的严肃探究最近才进入快车道。我们希望随着深度学习理论的进步，本书的每一版都能提供超越目前的见解。

为了避免不必要的重复，我们将一些最常导入和使用的函数及类封装在 `d2l` 包中。在全书中，我们用 `#@save`标记代码块（如函数、类或导入语句集合），表示它们稍后将通过 `d2l` 包被调用。我们在 :numref:`sec_d2l`中提供了这些类和函数的详细概述。 `d2l` 包是轻量级的，仅需要以下依赖项：

```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
本书中的大部分代码基于 Apache MXNet，这是一个用于深度学习的开源框架，是 AWS（亚马逊云服务）以及许多高校和公司的首选。本书中的所有代码均已在最新的 MXNet 版本下通过测试。但是，由于深度学习的快速发展，*印刷版中* 的某些代码在未来的 MXNet 版本中可能无法正常工作。我们计划保持在线版本更新。如果遇到任何问题，请查阅 :ref:`chap_installation` 以更新你的代码和运行环境。下面列出了我们的 MXNet 实现中的依赖项。
:end_tab:

:begin_tab:`pytorch`
本书中的大部分代码基于 PyTorch，这是一个流行的开源框架，受到了深度学习研究社区的热烈欢迎。本书中的所有代码均已在 PyTorch 的最新稳定版本下通过测试。但是，由于深度学习的快速发展，*印刷版中* 的某些代码在未来的 PyTorch 版本中可能无法正常工作。我们计划保持在线版本更新。如果遇到任何问题，请查阅 :ref:`chap_installation`以更新你的代码和运行环境。下面列出了我们的 PyTorch 实现中的依赖项。
:end_tab:

:begin_tab:`tensorflow`
本书中的大部分代码基于 TensorFlow，这是一个用于深度学习的开源框架，在工业界被广泛采用，在研究人员中也很受欢迎。本书中的所有代码均已在 TensorFlow 的最新稳定版本下通过测试。但是，由于深度学习的快速发展，*印刷版中* 的某些代码在未来的 TensorFlow 版本中可能无法正常工作。我们计划保持在线版本更新。如果遇到任何问题，请查阅 :ref:`chap_installation` 以更新你的代码和运行环境。下面列出了我们的 TensorFlow 实现中的依赖项。
:end_tab:

:begin_tab:`jax`
本书中的大部分代码基于 Jax，这是一个开源框架，支持可组合的函数变换，例如对任意 Python 和 NumPy 函数进行微分，以及 JIT 编译、向量化等等！它在机器学习研究领域正变得流行起来，并具有易于学习的类似 NumPy 的 API。实际上，JAX 试图实现与 NumPy 1:1 的对等兼容，因此切换代码可能就像更改单个导入语句一样简单！但是，由于深度学习的快速发展，*印刷版中* 的某些代码在未来的 Jax 版本中可能无法正常工作。我们计划保持在线版本更新。如果遇到任何问题，请查阅 :ref:`chap_installation`以更新你的代码和运行环境。下面列出了我们的 Jax 实现中的依赖项。
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab jax
#@save
from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any
```

### 目标读者

本书面向希望扎实掌握深度学习实践技术的学生（本科生或研究生）、工程师和研究人员。因为我们要从头开始解释每个概念，所以不需要先前的深度学习或机器学习背景。充分解释深度学习的方法需要一些数学和编程知识，但我们只假设你具备一些基础知识，包括适量的线性代数、微积分、概率论和 Python 编程。此外， [在线附录](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) 中，我们提供了本书所涵盖的大多数数学知识的复习。通常，我们会优先考虑直觉和思想，而不是数学上的严谨性。如果你想在理解本书的先决条件之外扩展这些基础，我们很高兴推荐一些其他极好的资源：:citet:`Bollobas.1999` 的 *Linear Analysis* 深入涵盖了线性代数和泛函分析；:cite:`Wasserman.2013`的 *All of Statistics* 提供了精彩的统计学介绍。 Joe Blitzstein 关于概率和推断的 [书籍](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918) 和 [课程](https://projects.iq.harvard.edu/stat110/home) 都是教学瑰宝。如果你以前没用过 Python，你可能想浏览一下这个 [Python教程](http://learnpython.org/).


### 笔记本、网站、GitHub 和论坛

我们可以从 [D2L.ai 网站](https://d2l.ai) 和 [GitHub](https://github.com/d2l-ai/d2l-en) 下载到所有的笔记本。配合本书，我们还在 [discuss.d2l.ai](https://discuss.d2l.ai/c/5) 推出了一个讨论论坛。每当且你有关于本书任何章节的问题时，你都可以在每个笔记本的末尾找到相关讨论页面的链接。


## 致谢

我们感谢数百位为英文版和中文版草稿做出贡献的人。他们帮助改进了内容并提供了宝贵的反馈。本书最初是以 MXNet 为主要框架实现的。我们感谢 Anirudh Dagar 和 Yuan Tang 分别将大部分早期的 MXNet 代码改编为 PyTorch 和 TensorFlow 实现。自 2021 年 7 月以来，我们使用 PyTorch、MXNet 和 TensorFlow 重新设计并重新实现了本书，并选择 PyTorch 作为主要框架。我们感谢 Anirudh Dagar 将大部分较新的 PyTorch 代码改编为 JAX 实现。我们感谢百度的 Gaosheng Wu, Liujun Hu, Ge Zhang, 和 Jiehang Xie 将中文草稿中大部分较新的 PyTorch 代码改编为 PaddlePaddle 实现。我们感谢 Shuai Zhang 将出版社的 LaTeX 样式整合到 PDF 构建中。

在 GitHub 上，我们感谢英文草稿的每一位贡献者，是你们让它对每个人都变得更好。他们的 GitHub ID 或名字如下（排名不分先后）： alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat, cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu, Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller, NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki, topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen, Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens, Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta, uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee, mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya, Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy, lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner, Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong, Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas, ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09, Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil, Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp, Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto, Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham, sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums, Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic, the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom, abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang, StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU, Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder, mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD, TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal, steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates, Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde, jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo, yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo, Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor, Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao, Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov, Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan, Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao, Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen, Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan, atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore, Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem, Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh, pggPL, RohanKarthikeyan, Ryan Choi, and Likun Lei.

我们感谢 Amazon Web Services，特别是 Wen-Ming Ye, George Karypis, Swami Sivasubramanian, Peter DeSantis, Adam Selipsky, 和 Andrew Jassy 对撰写本书的慷慨支持。如果没有可用的时间、资源、与同事的讨论以及持续的鼓励，本书就不可能完成。在本书准备出版期间，剑桥大学出版社提供了极好的支持。我们感谢我们的策划编辑 David Tranah 的帮助和专业精神。


## 摘要

深度学习彻底改变了模式识别，引入了现在为各种技术提供动力的技术，应用领域涵盖计算机视觉、自然语言处理和自动语音识别等。要成功应用深度学习，你必须了解如何定义问题、建模的基本数学原理、将模型拟合到数据的算法以及实现这一切的工程技术。本书提供了一个综合资源，将文字、图表、数学和代码集中在一处。


## 练习

1. 在本书的讨论论坛 [discuss.d2l.ai](https://discuss.d2l.ai/) 上注册一个帐户。
1. 在你的计算机上安装 Python。
1. 点击本节底部的论坛链接，你将能够寻求帮助、讨论本书内容，并通过与作者和更广泛的社区互动来找到问题的答案。

:begin_tab:`mxnet`
[讨论](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[讨论](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[讨论](https://discuss.d2l.ai/t/186)
:end_tab:

:begin_tab:`jax`
[讨论](https://discuss.d2l.ai/t/17963)
:end_tab:
