import os
import shutil


def preprocess_data_dirs(data_dirs, rewrite=False):
    flag = True
    for data_dir in data_dirs:
        flag = flag and os.path.exists(data_dir)
    
    for data_dir in data_dirs:
        if rewrite:
            shutil.rmtree(data_dir, ignore_errors=True)
        if len(os.path.basename(data_dir).split(".")) == 1:
            os.makedirs(data_dir, exist_ok=True)

    if rewrite:
        flag = False
    return flag