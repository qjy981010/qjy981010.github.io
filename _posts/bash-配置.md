---
title: bash 配置、技巧
date: 2017-12-22 20:58:40
tags: ['bash','bashrc','配置','技巧']
author: Jiyang Qi
---

作为Linux各大发行版的默认shell，使用bash是Linux使用者必备的技能

# Bash来历

bash为[GNU](https://zh.wikipedia.org/wiki/GNU)编写的一个命令处理器，他的名称是一个双关语，为Bourne-Again SHell或Born-Again SHell，其中[Bourne shell（sh）](https://zh.wikipedia.org/wiki/Bourne_shell)是之前Unix系统上常用的命令行处理器。之后bash一直沿用至今。

# bash 配置

一般我们的配置都写在`~/.bashrc`中，语法为bash语法， 此文件在每次登录或打开新的shell时执行，一般用来配置一些环境变量，定义函数，设置一些别名

我在这里贴一些通用配置，其他有关archlinux的一些专用配置可以参考[我的配置](https://github.com/qjy981010/vim-config/blob/master/.bashrc)

### 一堆别名
```bash
# alias命令可以设置别名
# alias <new cmd>='<origin cmd>'
# 这样以后想执行<origin cmd>时就可以直接执行更短更简单的<new cmd>了

# ls加颜色，通过颜色可辨别文件类型
alias ls='ls --color=auto'

# grep高亮，便于马上找到要找的字符串
alias grep='grep --color=auto' 
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# 获取自己的公网IP
alias publicip='curl ipinfo.io'

# 检查有没有网（不用可怜百度
alias netcheck='ping www.baidu.com'

# ll为人们约定俗成的全信息列举别名
alias ll='ls -lh'

# 外加列出隐藏目录
alias la='ls -lAh'

# 切换到上级目录，超实用
alias ..='cd ../'
alias ...='cd ../../'

# 过滤进程
alias pg='ps aux |grep -i'

# 过滤历史
alias hg='history |grep -i'

# 过滤当前目录文件
alias lg='ls -A |grep -i'

# 查看磁盘使用情况，主要是不加-h选项的话很不舒服，所以就设了个别名
alias df='df -Th'

# 查看内存、swap使用情况，-h同上
alias free='free -h'
```

### 环境变量设置
#### PATH
```bash
export PATH=$PATH:.
```
将当前目录加入PATH环境变量。$PATH：决定了shell将到哪些目录中寻找命令或程序，PATH的值是一系列目录，当您运行一个程序时，Linux在这些目录下进行搜寻编译链接。平时，若想运行此目录下的一个可执行文件，必须输入`./<file>`，但将当前目录加入PATH后，可直接执行`<file>`

#### EDITOR
```bash
export EDITOR="vim"
```
有些时候，大家可能被迫用vi（一个老式编辑器，不好用）

比如需要修改`/etc/sudoers`文件时，为了防止用户语法错误导致系统权限问题， 官方一般建议使用**visudo**这个命令，他默认使用vi作为编辑器。这时大家可以执行`export EDITOR="vim"`来设置EDITOR这个环境变量，再执行visudo，发现编辑器已经变成了自己熟悉的vim！

### 常用函数
```bash
# Useful unarchiver!
function extract () {
        if [ -f $1 ] ; then
                case $1 in
                        *.tar.bz2)        tar xjf $1                ;;
                        *.tar.gz)        tar xzf $1                ;;
                        *.bz2)                bunzip2 $1                ;;
                        *.rar)                rar x $1                ;;
                        *.gz)                gunzip $1                ;;
                        *.tar)                tar xf $1                ;;
                        *.tbz2)                tar xjf $1                ;;
                        *.tgz)                tar xzf $1                ;;
                        *.zip)                unzip $1                ;;
                        *.Z)                uncompress $1        ;;
                        *)                        echo "'$1' cannot be extracted via extract()" ;;
                esac
        else
                echo "'$1' is not a valid file"
        fi
}
```
以后解压文件直接`extract <file>`即可，超级实用

# bash 美化

不要委屈自己，装个powerline吧，用包管理工具安装我就不说了，下面是用pip装的步骤：

- 没有pip的同学先装pip
```bash
sudo apt-get install python-pip
```

- 安装powerline
```bash
$ pip install powerline-status
```

- powerline 中要用到特殊的字体，所以需要去下载字体
```bash
git clone https://github.com/powerline/fonts && ./fonts/install.sh
```

- 之后我们要每次启动bash时运行powerline，首先获取powerline的位置
```bash
$ pip show powerline-status
# 下面是我的输出
Name: powerline-status
Version: 2.6
Summary: The ultimate statusline/prompt utility.
Home-page: https://github.com/powerline/powerline
Author: Kim Silkebaekken
Author-email: kim.silkebaekken+vim@gmail.com
License: MIT
Location: /usr/lib/python3.6/site-packages
Requires:
```

- 可以看到我的Location是`/usr/lib/python3.6/site-packages/`，后面再加`powerline/bindings/bash/powerline.sh`就得到了我们要的powerline的位置。再在`.bashrc`中加一行，运行上面得到的**powerline.sh**就行了。我要加的是：
```bash
# 最前面的点是运行命令，与source命令相同
. /usr/lib/python3.6/site-packages/powerline/bindings/bash/powerline.sh
```

- 还要让你的终端支持256种颜色，还是在`~/.bashrc`里加一句：
```bash
export TERM="screen-256color"
```

# 其他shell
如果想尽可能提高效率，推荐大家使用**fish**这个shell，或者也可以用**zsh**，这里并不是说大家不用学bash了，因为bash到哪里都能用，但是另外两个不能，所以bash还是要学好。
### zsh
zsh是一个兼容bash，可扩展的shell，相当于一个给你配置好的bash。他还有很多插件，这里就不一一介绍了。推荐Oh-My-Zsh。
### 宇宙第一shell  fish
fish是目前比较强大的shell，他的缺点是不兼容bash，但他提供了更简单易用的语法和更完善的功能。不用配置就比装了许多插件的zsh更好用。  
同时他也有一些插件，也支持像bash那样的配置，配置文件在`~/.config/config.fish`。

**不建议**将fish设为默认shell，不然会很麻烦，在`~/.bashrc`里最后加一句`fish`就好。这样等你要用bash的时候直接`exit`就行了。

fish配置可以参考[这个博客](https://www.jianshu.com/p/7ffd9d1af788)