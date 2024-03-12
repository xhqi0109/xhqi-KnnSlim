'''
Author: Xiaohua000109 420889338@qq.com
Date: 2024-03-12 15:24:24
LastEditors: Xiaohua000109 420889338@qq.com
LastEditTime: 2024-03-12 15:29:08
FilePath: /undefined/Users/xhqi/Documents/xy/教程_备份/模型加速/xhqi-KnnSlim/setup.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from setuptools import setup, find_packages

setup(name='xhqi_knnslim',
      version='0.0.1',
      description='xhqi_knnslim model light weight Framework',
      author='xhqi_knnslim',
      requires=['numpy', 'matplotlib', 'tensorboard', 'torchaudio', 'torchvision', 'yaml'],  # 定义依赖哪些模块
      packages=['xhqi_knnslim'],
      license="apache 3.0"
      )
