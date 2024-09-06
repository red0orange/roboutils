import subprocess
import os
import sys
import signal
import atexit
import shutil

import open3d as o3d
import numpy as np
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program


class Open3dTBVis(object):
    def __init__(self, logdir="tmp_o3d_tb", remove_existing=False, launch=False):
        self.launch = launch
        self.logdir = logdir
        if remove_existing:
            shutil.rmtree(self.logdir, ignore_errors=True)

        self.writer = SummaryWriter(logdir)
        self.step = 0

        # @note 使得 step 会显示
        cache_pcd = np.ones([5, 3], dtype=np.float32) 
        o3d_cache_pcd = o3d.geometry.PointCloud()
        o3d_cache_pcd.points = o3d.utility.Vector3dVector(cache_pcd)
        self.add_3d("nouse_cache_1", o3d_cache_pcd)
        self.add_3d("nouse_cache_2", o3d_cache_pcd)

        # 直接启动会报错
        # if self.launch:
        #     self.tb = program.TensorBoard()
        #     self.tb.configure(argv=[None, '--logdir', os.path.abspath(self.logdir), '--port', '60008'])
        #     url = self.tb.launch()
        #     print(f"Tensorflow listening on {url}")
        pass

    def add_pcd(self, name, pcd):
        # @note tb 官方支持，但是不能多个点云显示在一个里面
        self.writer.add_mesh(name, pcd[None, ...], global_step=step)
        self.writer.flush()
        pass

    def add_3d(self, name, geometry, update_step=True):
        self.writer.add_3d(name, to_dict_batch([geometry]), self.step)
        # self.writer.flush()
        if update_step:
            self.step += 1
        else:
            pass

    def add_3ds(self, names, geometries, update_step=True):
        for name, geometry in zip(names, geometries):
            self.writer.add_3d(name, to_dict_batch([geometry]), self.step)
        # self.writer.flush()
        if update_step:
            self.step += 1
        else:
            pass

    def clear(self):
        # @note 不知道怎么 clear
        pass