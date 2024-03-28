import os
import glob
import numpy as np
from typing import Optional, List
import argparse
import json
from collections import defaultdict
import logging
import torch
import time
from tqdm import tqdm
from pathlib import Path
import open3d as o3d
import seaborn as sns
import matplotlib.pyplot as plt
from plot_statistics import SummaryKeys
from generate_statistics_summary_nuscenes import get_distance_str, perform_final_calculations, sort_summary

################################################################### 
bosch_cls= {
    0: 'PassengerCar',
    1: 'LargeVehicle',
    2: 'RidableVehicle',
    3: 'VulnerableRoadUser',
    }

bosch_list = [
    (0, 'Car'),
    (3, 'Pedestrian'),
    (2, 'Cyclist'),
    (2, 'rider'),
    (2, 'bicycle'),
    (-1, 'bicycle_rack'),
    (-1, 'human_depiction'),
    (2, 'moped_scooter'),
    (2, 'motor'),
    (2, 'ride_other'),
    (2, 'ride_uncertain'),
    (1, 'truck'),
    (1, 'vehicle_other'),
    ]

cls_to_idx = defaultdict(list)
for k, v in bosch_list:
    cls_to_idx[v.lower()] = k
print(f'cls_to_idx: {cls_to_idx}')

# def map_cls_to_idx(search_name):
#     print('search name: ', search_name.lower())
#     for idx, name in cls_to_idx.items():
#         print('find match: ', name.lower(), search_name.lower())
#         if name.lower() == search_name.lower():
#             return idx    
#     raise ValueError('search name not matched!')
###################################################################

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="Generate the statistics summary of the nuscenes dataset.")

    parser.add_argument(
        "--dataset_path",
        "-i",
        type=Path,
        help="Give a path to the dataset.",
        default='./distance_velocity_vod.npz',
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=False,
        help="Give a path to store the output json.",
        default='./',
    )

    args = parser.parse_args()
    # ####################################################################################################################################
    distances = np.arange(0, 200+1e-8, 50)
    velocities = np.arange(0, 50+1e-8, 0.2)
    npz = np.load(args.dataset_path, allow_pickle=True)
    hist_dist = npz['hist_dist']
    hist_velo = npz['hist_velo']
    print(f'histogram distance: {hist_dist.shape}, {hist_dist}')
    print(f'histogram velocity: {hist_velo.shape}, {hist_velo}')

    distance_bins = [f'{d[0]:.0f} - {d[1]:.0f} m' for d in list(zip(distances[:-1], distances[1:]))]
    print(f'distance bins: {len(distance_bins)}')

    velocity_bins = [f'{v[0]:.2f} - {v[1]:.2f} m/s' for v in list(zip(velocities[:-1], velocities[1:]))]
    print(f'velocity bins: {len(velocity_bins)}')

    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111)
    ax.bar(distance_bins, hist_dist[:, 1])
    ax.set_xlabel('distance')
    ax.set_ylabel('num of annotations')
    ax.set_title(f'histogram: distance')
    plt.suptitle('plot statistics')
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111)
    ax.bar(velocity_bins, hist_velo[:, 1])
    ax.set_xlabel('velocity')
    ax.set_ylabel('num of annotations')
    ax.set_title(f'histogram: velocity')
    plt.suptitle('plot statistics')
    plt.show()
    plt.close()




