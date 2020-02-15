from tqdm import tqdm
import sys
import os
from os import path
from keras.models import load_model
import cv2
import numpy as np
from helper import *
from keras.utils import multi_gpu_model
mod = load_model('mod.h5')
mod = multi_gpu_model(mod, gpus=8)

def get(path):
    from_mat = cv2.imread(path)
    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])
    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)
    return light_map


if __name__ == '__main__':
    batch_size = 100
    root = "/home/natsuki/danbooru2019"
    with open('list', 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    for j in tqdm(range(len(lines) // batch_size)):
        light_map = np.concatenate([
            get(path.join(root, '512px', line))
            for line in lines[j * batch_size:(j + 1) * batch_size]
        ])
        output = mod.predict(light_map)
        for i in range(batch_size):
            line_mat = output[3 * i:3 * (i + 1)].transpose((3, 1, 2, 0))[0]
            line = lines[j*batch_size+i]
            show_active_img_and_save('', line_mat, path.join(root, 'sketchKeras_colored', line))
            line_mat = np.amax(line_mat, 2)
            show_active_img_and_save_denoise_filter2('', line_mat, path.join(root, 'sketchKeras_enhanced', line))
            show_active_img_and_save_denoise_filter('', line_mat, path.join(root, 'sketchKeras_pured', line))
            show_active_img_and_save_denoise('', line_mat, path.join(root, 'sketchKeras', line))