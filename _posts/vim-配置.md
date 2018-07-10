---
title: vim 常用技巧、配置、插件
date: 2017-12-22 18:19:22
tags: ["vim","配置","技巧","插件","vimrc"]
author: Jiyang Qi
subtitle: 打造你的终极vim
---
作为命令行界面下的首要编辑器，vim虽然不如各大IDE强大，但要想优雅地使用Linux，必须先熟练的使用vim。他有以下几个优点：

* 能方便的在命令行下编辑查看文件，易实现无鼠标操作，甚至不需要去按方向键就可以实现方便的光标移动，对于广大工作在一线的程序员，这无疑是提高生产效率的利器。不然，为什么各大IDE里都会有将自己**vim化**的的插件呢。
* 定制性强；尽管在许多人眼里，vim和IDE根本不是同一个级别，但只要经过简单的配置，你一定会爱上这个小东西。他可以通过配置文件配置，还有众多开源插件，足以追上IDE甚至在某些方面超过他们。

## 技巧
### 以root权限写入

你有没有遇到过这种情况：当你修改一个系统文件时，要保存时突然发现忘记加`sudo`命令了，自己没有**写入权限**。这时如果关掉文件再重新打开，前面写的东西岂不是都白写了？。不用担心，在命令模式下输入`:w !sudo tee %`，回车，`L`。或者更舒服一点`:w !sudo tee > /dev/null %`，回车。

其中`:w`把当前文件内容输入到标准输入，这时文件内容就在输入缓冲区中了。`！{cmd}`表示调用外部命令`{cmd}`，这里的外部命令就是`sudo tee %`了，`sudo`获取root权限，`tee`命令将输入缓冲区内容输入到文件`%`，vim里的`%`就代表当前文件名。

于是你刚刚的修改就被巧妙的以root权限写下了。
### EDITOR环境变量
有些时候，大家可能被迫用vi（一个老式编辑器，不好用）

比如需要修改`/etc/sudoers`文件时，为了防止用户语法错误导致系统权限问题， 官方一般建议使用**visudo**这个命令，他默认使用vi作为编辑器。这时大家可以执行`export EDITOR="vim"`来设置EDITOR这个环境变量，再执行visudo，发现编辑器已经变成了自己熟悉的vim！

### 实用vim命令
- `vsp <file>`水平分屏并在新窗口打开<file>，对应的还有`sp <file>`垂直分屏  
- `G`跳到文件结尾，`gg`跳到文件开头  
- `j`、`k`、`h`、`l`方便的移动光标，妈妈再也不用担心我键盘上没有方向键啦～  
- `/<你要搜索的字符串>`字符串搜索，相当于Ctrl-F  
- `yy`复制当前行，`<n>yy`复制当前行及下面的`n-1`行
- `dd`剪切当前行，`<n>dd`剪切当前行及下面的`n-1`行
- `p`粘贴

### chrome Vimium插件
无鼠标查网页，操作chrome，键盘党福利。

## 配置
vim的配置文件为`~/.vimrc`  
这里贴一些常用配置，更多可以参考[我的配置](https://github.com/qjy981010/vim-config/blob/master/.vimrc)
```
""""""""""""""""""""""""" 常用快捷键 """"""""""""""""""""""""""""""

" 双引号在vim配置文件中代表注释
" Ctrl+A 全选
map <C-A> ggVG

" Ctrl+C 全局复制 （对archlinux系统不能复制到全局剪切板）
vnoremap <C-c> "+y
"archlinux参考http://blog.fooleap.org/using-vim-with-clipboard-on-archlinux.html
"不过archlinux官方源里现在已经没有ABS了

" Ctrl+v 粘贴
map <C-v> "+p
imap <C-v> <esc>"+pa
vmap <C-v> d"+p

""""""""""""""""""""""""""""" 代码必需 """"""""""""""""""""""""""""""

" 语法高亮 
syntax enable
syntax on 

"显示行号
set number

" 当前行，行号高亮
highlight clear LineNr

" 继承前一行的缩进方式，特别适用于多行注释 
set autoindent 

" 为C程序提供自动缩进 
set smartindent 

" 使用C样式的缩进 
set cindent 

" 制表符为4 
set tabstop=4 

" 统一缩进为4 
set softtabstop=4 
set shiftwidth=4 

" python 用空格代替制表符 
autocmd FileType python set expandtab 

"突出当前行
set cursorline
hi CursorLine   cterm=NONE ctermbg=black ctermfg=NONE guibg=NONE guifg=NONE

"智能补全
set completeopt=longest,menu

"括号补全
:inoremap ( ()<ESC>i
:inoremap { {}<ESC>i
:inoremap [ []<ESC>i
:inoremap " ""<ESC>i
:inoremap ' ''<ESC>i
:inoremap ` ``<ESC>i

" 可以在buffer的任何地方使用鼠标（类似office中在工作区双击鼠标定位） 
set mouse=a 
set selection=exclusive 
set selectmode=mouse,key 

" 高亮显示匹配的括号 
set showmatch 

" 匹配括号高亮的时间（单位是十分之一秒） 
set matchtime=0

" 在搜索的时候忽略大小写 
set ignorecase 

" 不要高亮被搜索的句子（phrases） 
set nohlsearch 

" 光标移动到buffer的顶部和底部时保持3行距离 
set scrolloff=3 

" 共享剪切板
set clipboard+=unnamed

" 不为正在编辑的文件生成swap文件
set noswapfile

" 设置当文件被改动时自动载入
set autoread

" quickfix模式
autocmd FileType c,cpp map <buffer> <leader><space> :w<cr>:make<cr>

" 启动的时候不显示那个援助索马里儿童的提示 
set shortmess=atI 

" 不让vim发出讨厌的滴滴声 
set noerrorbells


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""" 高能预警 """""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""""""""""""""""""""""""""" 自动插入文件头 """"""""""""""""""""""""""""""""""

"新建.c,.h,.sh,.java文件，自动插入文件头 
autocmd BufNewFile *.cpp,*.[ch],*.sh,*.rb,*.java,*.py exec ":call SetTitle()" 
""定义函数SetTitle，自动插入文件头 
func SetTitle() 
    "如果文件类型为.sh文件 
    if &filetype == 'sh' 
        call setline(1,"\#!/bin/bash") 
        " call append(line("."), "") 
    elseif &filetype == 'python'
        call setline(1,"#!/usr/bin/env python")
        call append(line("."),"# coding=utf-8")
        call append(line(".")+1, "") 

    elseif &filetype == 'ruby'
        call setline(1,"#!/usr/bin/env ruby")
        call append(line("."),"# encoding: utf-8")
        call append(line(".")+1, "")

"    elseif &filetype == 'mkd'
"        call setline(1,"<head><meta charset=\"UTF-8\"></head>")
    
    endif
    if expand("%:e") == 'cpp'
        call setline(1, "#include<iostream>")
        call append(line("."), "")
        call append(line(".")+1, "using namespace std;")
        call append(line(".")+2, "")
    endif
    if &filetype == 'c'
        call setline(1, "#include<stdio.h>")
        call append(line("."), "")
    endif
    if expand("%:e") == 'h'
        call setline(1, "#ifndef ".toupper(expand("%:r"))."_H_")
        call append(line("."), "#define ".toupper(expand("%:r"))."_H_")
        call append(line(".")+1, "#endif")
    endif
    if &filetype == 'java'
        call setline(1,"public class ".expand("%:r"))
        call append(line("."),"")
    endif
    "新建文件后，自动定位到文件末尾
endfunc 
autocmd BufNewFile * normal G

"""""""""""""""""""""""""" 快捷编译运行 """"""""""""""""""""""""""""""

" from https://github.com/ma6174/vim 
" 按F5编译运行
map <F5> :call CompileRunGcc()<CR>
func! CompileRunGcc()
    exec "w"
    if &filetype == 'c'
        exec "!g++ % -o %<"
        exec "!time ./%<"
    elseif &filetype == 'cpp'
        exec "!g++ % -std=c++11 -o %<"
        exec "!time ./%<"
    elseif &filetype == 'java' 
        exec "!javac %" 
        exec "!time java %<"
    elseif &filetype == 'sh'
        :!time bash %
    elseif &filetype == 'python'
        exec "!time python %"
    elseif &filetype == 'html'
        exec "!firefox % &"
    elseif &filetype == 'go'
"        exec "!go build %<"
        exec "!time go run %"
    elseif &filetype == 'mkd'
        exec "!~/.vim/markdown.pl % > %.html &"
        exec "!firefox %.html &"
    endif
endfunc

"C,C++的调试
map <F8> :call Rungdb()<CR>
func! Rungdb()
    exec "w"
    exec "!g++ % -g -o %<"
    exec "!gdb ./%<"
endfunc

"代码格式优化化
map <F6> :call FormartSrc()<CR><CR>
"定义FormartSrc()
func FormartSrc()
    exec "w"
    if &filetype == 'c'
        exec "!astyle --style=ansi -a --suffix=none %"
    elseif &filetype == 'cpp' || &filetype == 'hpp'
        exec "r !astyle --style=ansi --one-line=keep-statements -a --suffix=none %> /dev/null 2>&1"
    elseif &filetype == 'perl'
        exec "!astyle --style=gnu --suffix=none %"
    elseif &filetype == 'py'||&filetype == 'python'
        exec "r !autopep8 -i --aggressive %"
    elseif &filetype == 'java'
        exec "!astyle --style=java --suffix=none %"
    elseif &filetype == 'jsp'
        exec "!astyle --style=gnu --suffix=none %"
    elseif &filetype == 'xml'
        exec "!astyle --style=gnu --suffix=none %"
    else
        exec "normal gg=G"
        return
    endif
    exec "e! %"
endfunc
"结束定义FormartSrc

```

想速成的选手可以尝试一下这个[github上开源的比较完善的配置](https://github.com/amix/vimrc)，但是毕竟是别人的配置，关键是要明白每个配置的效果，不然，可能就算有这个配置，大家也不会用这些功能。所以建议大家可以多尝试一下，尽量自己探索，**适合你的配置才是最好的配置**。

## 插件推荐
以下插件应该均可通过Linux各发行版的包管理工具下载，也可以通过Vundle安装

- [Vundle](https://github.com/VundleVim/Vundle.vim)：插件安装利器
- [YouCompleteMe](https://github.com/Valloric/YouCompleteMe)智能补全及错误提示，想配置好略麻烦，但是好用
- [syntastic](https://github.com/vim-syntastic/syntastic)：语法检查
- [Powerline](https://github.com/Lokaltog/vim-powerline) / [Airline](https://github.com/vim-airline/vim-airline)：美化工具，养眼
- [NerdTree](https://github.com/scrooloose/nerdtree)：文件管理，多文件切换
- [UltiSnips](https://github.com/SirVer/ultisnips)：代码块补全，码农必备、、

**友情提示**：插件不易装太多，不然你的vim，启动会变慢，导致体验 - -

**最后，大家就可以愉快而优雅的使用vim了**