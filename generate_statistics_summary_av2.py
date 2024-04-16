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
from kornia.geometry.linalg import transform_points
from av2.torch.data_loaders.detection import DetectionDataLoader
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.utils.typing import NDArrayByte, NDArrayInt
from av2.rendering.color import ColorFormats, create_range_map
from av2.rendering.rasterize import draw_points_xy_in_img
import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# mp.set_start_method('fork')

################################################################### 
bosch_cls= {
    0: 'PassengerCar',
    1: 'LargeVehicle',
    2: 'RidableVehicle',
    3: 'VulnerableRoadUser',
    }

bosch_list = [
    (0, 'REGULAR_VEHICLE'),
    (3, 'PEDESTRIAN'),
    (-1, 'BOLLARD'),
    (-1, 'CONSTRUCTION_CONE'),
    (-1, 'CONSTRUCTION_BARREL'),
    (-1, 'STOP_SIGN'),
    (2, 'BICYCLE'),
    (1, 'LARGE_VEHICLE'),
    (2, 'WHEELED_DEVICE'),
    (1, 'BUS'),
    (1, 'BOX_TRUCK'),
    (-1, 'SIGN'),
    (1, 'TRUCK'),
    (2, 'MOTORCYCLE'),
    (2, 'BICYCLIST'),
    (1, 'VEHICULAR_TRAILER'),
    (1, 'TRUCK_CAB'),
    (2, 'MOTORCYCLIST'),
    (1, 'DOG'),
    (1, 'SCHOOL_BUS'),
    (2, 'WHEELED_RIDER'),
    (2, 'STROLLER'),
    (1, 'ARTICULATED_BUS'),
    (1, 'MESSAGE_BOARD_TRAILER'),
    (-1, 'MOBILE_PEDESTRIAN_SIGN'),
    (-1, 'MOBILE_PEDESTRIAN_CROSSING_SIGN'),
    (2, 'WHEELCHAIR'),
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

class AV2_dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, dataset_name='Argoverse2', split_name='val', timestep=1):
        self.timestep = timestep
        self.detection_loader = DetectionDataLoader(
            root_dir,
            dataset_name,
            split_name,
            num_accumulated_sweeps=1,
        )
        print('length of the detection lodaer: ', len(self.detection_loader))
        self.sensor_loader = SensorDataloader(Path(os.path.join(root_dir, dataset_name, 'sensor')), 
                                              with_annotations=False, 
                                              with_cache=True) 
        print('length of the sensor lodaer: ', len(self.sensor_loader))

    def transform_points(self, points, transformation):
        points_h = np.concatenate((points, np.ones((len(points), 1))), axis=1)
        points_t = transformation @ points_h.T
        return points_t.T[:, 0:3]

    def is_in_bbox(self, points, corners):
        v1 = corners[1, 0:3] - corners[0, 0:3]
        v2 = corners[3, 0:3] - corners[0, 0:3]
        v3 = corners[4, 0:3] - corners[0, 0:3]
        vs = points - corners[0:1, 0:3]
        v1s = np.dot(vs, v1)
        v2s = np.dot(vs, v2)
        v3s = np.dot(vs, v3)
        # print('corners: ', corners)
        x_in =np.logical_and(v1s>=0, v1s<np.dot(v1, v1))
        y_in =np.logical_and(v2s>=0, v2s<np.dot(v2, v2))
        z_in =np.logical_and(v3s>=0, v3s<np.dot(v3, v3))
        is_in = np.logical_and(x_in, y_in, z_in)
        return is_in, np.sum(is_in)

    def get_3d_corners(self, xyzlwh):
        x = xyzlwh[:, 0]
        y = xyzlwh[:, 1]
        z = xyzlwh[:, 2]

        l = xyzlwh[:, 3]
        w = xyzlwh[:, 4]
        h = xyzlwh[:, 5]
        x_corners = [l / 2,
                     l / 2,
                     -l / 2,
                     -l / 2,
                     l / 2,
                     l / 2,
                     -l / 2,
                     -l / 2]
        y_corners = [w / 2,
                     -w / 2,
                     -w / 2,
                     w / 2,
                     w / 2,
                     -w / 2,
                     -w / 2,
                     w / 2]
        z_corners = [np.zeros((len(h))),
                     np.zeros((len(h))),
                     np.zeros((len(h))),
                     np.zeros((len(h))),
                     h,
                     h,
                     h,
                     h]
        x_corners = np.stack(x_corners, axis=1)
        y_corners = np.stack(y_corners, axis=1)
        z_corners = np.stack(z_corners, axis=1)
        corners_3d = np.stack([x_corners, y_corners, z_corners], axis=-1)
        return corners_3d + xyzlwh[:, 0:3][:, None, :]

    def load_frame(self, index):
        sweep = self.detection_loader.__getitem__(index)
        _, timestamp = sweep.sweep_uuid
        # Lidar (x,y,z) in meters and intensity (i).
        lidar = sweep.lidar.as_tensor().numpy()
        # visualize_pcd(lidar[:, 0:3])

        # # 4x4 matrix representing the SE(3) transformation to city from ego-vehicle coordinates.
        city_SE3_ego_mat4 = sweep.city_SE3_ego.matrix().numpy()[0]

        # # Transform the points to city coordinates.
        # lidar_xyz_city = transform_points(city_SE3_ego_mat4, lidar_tensor[:, :3])

        bboxes = []
        # Cuboids might not be available (e.g., using the "test" split).
        if sweep.cuboids is not None:
            # Annotations in (x,y,z,l,w,h,yaw) format.
            # 1-DOF rotation.
            xyzlwh_t = sweep.cuboids.as_tensor().numpy()
            corners = self.get_3d_corners(xyzlwh_t)
            # print('corners: ', type(corners), corners.shape)

            # Access cuboid category.
            category = sweep.cuboids.category
            # print('category: ', type(category), len(category))

            # Access track uuid.
            track_uuid = sweep.cuboids.track_uuid
            # print('track uuid: ', type(track_uuid), len(track_uuid))
            # print(type(xyzlwh_t), xyzlwh_t, category, track_uuid)

            # calculate inliers
            assert len(xyzlwh_t)==len(corners)
            assert len(xyzlwh_t)==len(category)
            assert len(xyzlwh_t)==len(track_uuid)
            for k in range(0, len(xyzlwh_t)): 
                valid, _ = self.is_in_bbox(lidar[:, 0:3], corners[k])
                label = np.zeros((len(lidar)))
                label[valid] = 1
                # self.visualize_pcd(lidar, label, title=f'id: {track_uuid[k]}, {category[k]}, num of lidar points: {sum(valid)}')

                bboxes.append({'label_class': category[k],
                                'track_id': track_uuid[k],
                                'dimensions': xyzlwh_t[k, 3:6], 
                                'corners_3d': corners[k], 
                                'center_3d': xyzlwh_t[k, 0:3],  
                                'valid_lidar': valid,
                                })

        return lidar, timestamp, city_SE3_ego_mat4, bboxes

    def compute_velocity(self, bboxes1, bboxes2, transforms1, transforms2, timestep):
        for i, bbox1 in enumerate(bboxes1):
            id1 = bbox1['track_id']
            pos1 = bbox1['center_3d']
            pos1_city = self.transform_points(pos1[None, :], transforms1)

            tracked = False
            for j, bbox2 in enumerate(bboxes2):
                id2 = bbox2['track_id']
                if id1 == id2: 
                    tracked = True
                    break
                else: 
                    pass
            # print('id1, id2: ', j, id1, id2, len(bboxes2))
            if tracked:
                pos2 = bboxes2[j]['center_3d']
                pos2_city = self.transform_points(pos2[None, :], transforms2)
                velocity = (pos2_city - pos1_city)[0, 0:3] / timestep 
                bbox1['velocity'] = velocity
                # print(f"compute velocity: {bbox1['velocity']}, {np.linalg.norm(velocity)}, {np.linalg.norm(pos2_odom[0, 0:3] - pos1_odom[0, 0:3]) / timestep}, {timestep}")
            else:
                # print('No track id!')
                bbox1['velocity'] = np.zeros((3))-1e8
        return bboxes1

    #############################################################################
    # # # visualize
    def visualize_pcd(self, points, labels=None, num_colors=2, title='visualization'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        if labels is None:
            pass
        else:
            COLOR_MAP = sns.color_palette('husl', n_colors=num_colors)
            COLOR_MAP = np.array(COLOR_MAP)
            # print('COLOR_MAP: ', COLOR_MAP)
            labels = labels.astype(int)
            colors = COLOR_MAP[labels%len(COLOR_MAP)]
            colors[labels < 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name=title)

    ########################################################################################
    # # # Pytorch dataloader
    def __len__(self):
        # print('total samples: ', len(self.detection_loader))
        return len(self.detection_loader)

    def __getitem__(self, idx):
        idx1 = idx
        idx2 = idx1+self.timestep if idx1+self.timestep<self.__len__() else idx1-self.timestep
        # print('idx: ', idx1, idx2)
        # timestep = np.abs(idx2-idx1) * 0.1
        lidar1, timestamp1, transforms1, bboxes1 = self.load_frame(idx1)
        lidar2, timestamp2, transforms2, bboxes2 = self.load_frame(idx2)
        timestep = np.abs(timestamp2 - timestamp1) * 1e-9
        # print('timestamp: ', timestamp1, timestamp2, timestep)
        bboxes1 = self.compute_velocity(bboxes1, bboxes2, transforms1, transforms2, timestep)

        # ############################################################################
        # # # debug
        # lidar1_city = self.transform_points(lidar1[:, 0:3], transforms1)
        # lidar2_city = self.transform_points(lidar2[:, 0:3], transforms2)
        # # self.visualize_pcd(np.concatenate([lidar1_city, lidar2_city], axis=0)[:, 0:3],  np.hstack([np.zeros((len(lidar1))), np.ones((len(lidar2)))]), title=f'ego motion')
        # lidar = np.concatenate([lidar1_city, lidar2_city], axis=0)[:, 0:3]

        # for bbox1 in bboxes1:
        #     # # if bbox1['label_class'] == 'Pedestrian': 
        #     # if bbox1['track_id'] == 1516: 
        #     #     pass
        #     # else:
        #     #     continue
        #     valid_lidar = bbox1['valid_lidar']
        #     velocity = np.linalg.norm(bbox1['velocity'][0:2])
        #     label = np.hstack([np.zeros((len(lidar1)))+1, np.zeros((len(lidar2)))+2])
        #     label[np.flatnonzero(valid_lidar)] = 0
        #     self.visualize_pcd(lidar[:, 0:3], label, num_colors=3,
        #                    title=f"{idx}, id: {bbox1['track_id']}, {bbox1['label_class']}, lidar points: {sum(valid_lidar)}, velocity: {velocity} m/s")

        # # # array of center_3d and velocity 
        cls_labels = []
        dists = []
        velos = []
        nums_lidar = []
        areas = []
        
        for bbox1 in bboxes1:
            # print('bbox1 label: ', bbox1['label_class'], cls_to_idx[bbox1['label_class'].lower()])
            # there are some outlier class names
            cls_label = cls_to_idx[bbox1['label_class'].lower()]
            if cls_label ==[]: 
                cls_label = -1
            cls_labels.append( cls_label)
            dists.append(np.linalg.norm(bbox1['center_3d'][0:2]))
            velos.append(np.linalg.norm(bbox1['velocity'][0:2]))
            nums_lidar.append(sum(bbox1['valid_lidar']))
            areas.append(np.prod(bbox1['dimensions'][0:2]))
        return lidar1, np.array(cls_labels), np.array(dists), np.array(velos), np.array(nums_lidar), np.array(areas)

# https://discuss.pytorch.org/t/is-a-dataset-copied-as-part-of-dataloader-with-multiple-workers/112924
def collate(batch):
    # print('collate batch: ', len(batch), len(batch[0]))
    return batch

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Generate the statistics summary of the nuscenes dataset.")
    parser.add_argument(
        "--dataset_path",
        "-i",
        type=Path,
        help="Give a path to the dataset.",
        default='/media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset',
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
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = AV2_dataset(root_dir=args.dataset_path, split_name='train', timestep=1)

    kwargs = {
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 0, 
        "pin_memory": False,
        "collate_fn": collate,
    }
    data_loader = torch.utils.data.DataLoader(
        dataset,
        **kwargs,
    )

    ####################################################################################################################################
    ### HISTOGRAM
    distances = np.arange(0, 200+1e-8, 1.0)
    velocities = np.arange(0, 50+1e-8, 0.2)

    hist_dist = np.zeros((len(distances)-1, 2))
    hist_velo = np.zeros((len(velocities)-1, 2))
    hist_dist[:, 0] = distances[0:-1]
    hist_velo[:, 0] = velocities[0:-1]

    for i, batch in enumerate(tqdm(data_loader)):
        lidar, labels, dists, velos, nums_lidar, areas = batch[0]

        bin_dist, edge_dist = np.histogram(dists, distances)
        # print('edge_dist: ', edge_dist.shape, edge_dist)
        hist_dist[:, 1] += bin_dist

        bin_velo, edge_velo = np.histogram(velos, velocities)
        # print('edge_velo: ', edge_velo, edge_velo.max(), edge_velo.min())
        hist_velo[:, 1] += bin_velo
    np.savez(args.output.joinpath("distance_velocity_av2.npz"), hist_dist = hist_dist, hist_velo = hist_velo, dist_bins=distances, velo_bins=velocities)    
    ###################################################################################################################################
    ### POINT CLOUD ANALYSIS
    distances = np.arange(0, 200+1e-8, 50)
    distance_bins = list(zip(distances[:-1], distances[1:]))
    # print('distances: ', distances, distance_bins)

    structured_summary = {"name": "av2"}
    structured_summary["box_results"] = defaultdict(
        lambda: defaultdict(
            lambda: {
                SummaryKeys.TOTAL_POINTS_RADAR.full_name: 0,
                SummaryKeys.TOTAL_POINTS_LIDAR.full_name: 0,
                SummaryKeys.TOTAL_AREA.full_name: 0,
                SummaryKeys.TOTAL_NO_BOXES.full_name: 0,
            }
        )
    )
    structured_summary["frame_results"] = defaultdict(
        lambda: {SummaryKeys.TOTAL_POINTS_RADAR.full_name: 0, SummaryKeys.TOTAL_POINTS_LIDAR.full_name: 0}
    )
    structured_summary[SummaryKeys.TOTAL_NO_FRAMES.full_name] = 0

    for i, batch in enumerate(tqdm(data_loader)):
        assert len(batch)==1
        lidar, labels, dists, velos, nums_lidar, areas = batch[0]
        structured_summary[SummaryKeys.TOTAL_NO_FRAMES.full_name] += len(batch)
        # print('var size: ', img.shape, lidar.shape, radar.shape, labels.shape, dists.shape, velos.shape, nums_lidar.shape, areas.shape)
        # per box
        for label, dist, velo, num_lidar, area in zip(labels, dists, velos, nums_lidar, areas):
            # print('img, lidar, radar: ', img.shape, lidar.shape, radar.shape)
            if int(label)<0: continue
            label, dist, velo, num_lidar, area = int(label), float(dist), float(velo), int(num_lidar), float(area)
            class_name = bosch_cls[label]

            distance_bin_str = get_distance_str(dist, distance_bins)
            if distance_bin_str is None:
                continue

            structured_summary["box_results"][class_name][distance_bin_str][
                SummaryKeys.TOTAL_POINTS_RADAR.full_name
            ] += 0
            structured_summary["box_results"][class_name][distance_bin_str][
                SummaryKeys.TOTAL_POINTS_LIDAR.full_name
            ] += num_lidar
            structured_summary["box_results"][class_name][distance_bin_str][
                SummaryKeys.TOTAL_AREA.full_name
            ] += area
            structured_summary["box_results"][class_name][distance_bin_str][
                SummaryKeys.TOTAL_NO_BOXES.full_name
            ] += 1

        # per frame
        bins_lidar, edges_dist = np.histogram(np.linalg.norm(lidar[:, 0:2], axis=1), distances)
        # print('edges_dist: ', bins_lidar.shape, edges_dist.shape, edges_dist)

        for k, edge_dist in enumerate(edges_dist[0:-1]): 
            dis_bin_str = get_distance_str(edge_dist, distance_bins)
            structured_summary["frame_results"][distance_bin_str][
                SummaryKeys.TOTAL_POINTS_LIDAR.full_name
            ] += int(bins_lidar[k])
            structured_summary["frame_results"][distance_bin_str][
                SummaryKeys.TOTAL_POINTS_RADAR.full_name
            ] += 0

    structured_summary = perform_final_calculations(structured_summary)
    structured_summary = sort_summary(structured_summary)

    # save summary to json
    with open(args.output.joinpath("summary_av2.json"), "w", encoding="utf-8") as handle:
        json.dump(structured_summary, handle, indent=2)

    print(f"Saved summary_nuscenes.vod to {args.output.absolute()}")
