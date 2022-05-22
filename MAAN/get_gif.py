import imageio
import os

'''
# 只支持png格式，需要先命名排序好(默认按照字母序排列)
# source(字符串)：素材图片路径，生成的gif也保存在该路径
# gifname(字符串)：生成的gif的文件名，命名时带后缀如：'1.gif'
# time(数字)：生成的gif每一帧的时间间隔，单位（s）
'''


def path_valid(path):
    if not os.path.exists(path):
        os.mkdir(path)


def png2gif(source, dir_name, gifname, time):
    # os.chdir(source)  # os.chdir()：改变当前工作目录到指定的路径
    file_list = os.listdir(source)  # os.listdir()：文件夹中的文件/文件夹的名字列表
    frames = []  # 读入缓冲区
    for png in file_list:
        png_ = os.path.join(source, png)
        frames.append(imageio.imread(png_))
    gifname = os.path.join(gifname, dir_name + '.gif')
    imageio.mimsave(gifname, frames, 'GIF', duration=time)


def  make_gif(address, gif_path):
    # address = "./test_0403"
    # gif_path = './gifs_0403'
    path_valid(gif_path)
    l = []
    for dir_name in os.listdir(address):
        l.append(int(dir_name))
        path_ = os.path.join(address, dir_name)
        png2gif(path_,  dir_name, gif_path, 0.1)
