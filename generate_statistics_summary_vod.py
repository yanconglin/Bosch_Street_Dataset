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
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, FrameLabels, transform_pcl, project_pcl_to_image, min_max_filter, homogeneous_transformation, homogeneous_coordinates
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

class VOD_dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, timestep=1):
        self.root_dir = root_dir
        self.timestep = timestep
        # self.frame_list = [ idx.replace('.jpg', '')  for idx in os.listdir(os.path.join(self.camera_dir))]
        with open(os.path.join(self.root_dir, 'lidar', 'ImageSets', 'train.txt')) as file:
            self.frame_list = [line.rstrip() for line in file]
            # self.frame_list = self.frame_list[0:100]
            # for line in file:
            #     if '01313' in line: pass # bad sample
            #     else:self.frame_list.append(line.rstrip())
        print('total nbr of frames: ', len(self.frame_list))

    def get_labels_dict(self, raw_labels) -> List[dict]:
        labels = []  # List to be filled
        for act_line in raw_labels:  # Go line by line to split the keys
            act_line = act_line.split()
            label, id, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
            h, w, l, x, y, z, rot, score = map(float, [h, w, l, x, y, z, rot, score])
            labels.append({'label_class': label,
                           'track_id': int(id),
                           'h': h,
                           'w': w,
                           'l': l,
                           'x': x,
                           'y': y,
                           'z': z,
                           'rotation': rot,
                           'score': score}
                          )
        return labels

    def get_3d_corners(self, label):
        x_corners = [label['l'] / 2,
                     label['l'] / 2,
                     -label['l'] / 2,
                     -label['l'] / 2,
                     label['l'] / 2,
                     label['l'] / 2,
                     -label['l'] / 2,
                     -label['l'] / 2]
        y_corners = [label['w'] / 2,
                     -label['w'] / 2,
                     -label['w'] / 2,
                     label['w'] / 2,
                     label['w'] / 2,
                     -label['w'] / 2,
                     -label['w'] / 2,
                     label['w'] / 2]
        z_corners = [0,
                     0,
                     0,
                     0,
                     label['h'],
                     label['h'],
                     label['h'],
                     label['h']]

        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        return corners_3d

    def is_in_bbox(self, points, corners):
        # bbox is rotated!
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

    def load_frame(self, idx):
        # print('idx: ', idx, self.frame_list[idx])
        kitti_locations = KittiLocations(root_dir=self.root_dir,
                                        output_dir="",
                                        frame_set_path="",
                                        pred_dir="",
                                        )
        frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=self.frame_list[idx])
        transforms = FrameTransformMatrix(frame_data)

        image = frame_data.get_image()
        lidar = frame_data.get_lidar_scan() 
        radar = frame_data.get_radar_scan() 

        # lidar_show = np.concatenate([lidar[:, 0:3], np.zeros((1,3))], axis=0)
        # label_show = np.concatenate([np.zeros((len(lidar))), np.array([1])], axis=0)
        # self.visualize_pcd(lidar_show, label_show, title='visualize lidar')

        label = frame_data.get_labels()
        label = self.get_labels_dict(label)
        bboxes = self.get_bboxes(image, lidar, radar, label, transforms)
        return image, lidar, radar, transforms, bboxes


    def get_bboxes(self, image, lidar, radar, label_list, transforms):
        bboxes = []
        for index, label in enumerate(label_list):
            rotation = -(label['rotation'] + np.pi / 2)  # undo changes made to rotation
            rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                                   [np.sin(rotation), np.cos(rotation), 0],
                                   [0, 0, 1]])

            # center = (np.linalg.inv(transforms.t_camera_lidar) @ np.array([label['x'],
            #                                                      label['y'],
            #                                                      label['z'],
            #                                                      1]))
            center = (transforms.t_lidar_camera @ np.array([label['x'],
                                                                 label['y'],
                                                                 label['z'],
                                                                 1]))
            corners_3d = self.get_3d_corners(label)
            new_corners_3d = np.dot(rot_matrix, corners_3d).T + center[:3]
            # map bboxes to camera frame
            new_corners_3d_cam = homogeneous_transformation(homogeneous_coordinates(new_corners_3d), transforms.t_camera_lidar)

            # map lidar and radar points to camera frame
            lidar_cam = homogeneous_transformation(homogeneous_coordinates(lidar[:, 0:3]), transforms.t_camera_lidar)[:, 0:3]
            radar_cam = homogeneous_transformation(homogeneous_coordinates(radar[:, 0:3]), transforms.t_camera_radar)[:, 0:3]

            valid_lidar, count_lidar = self.is_in_bbox(lidar_cam[:, 0:3], new_corners_3d_cam)
            valid_radar, count_radar = self.is_in_bbox(radar_cam[:, 0:3], new_corners_3d_cam)
            if not (sum(valid_lidar)>0 or sum(valid_radar)>0): continue
            # self.visualize_pcd(lidar[:, 0:3], valid_lidar, title=f'lidar_{count_lidar}')
            # self.visualize_pcd(radar[:, 0:3], valid_radar, title=f'radar_{count_radar}')
            # self.visualize_all(lidar[valid_lidar], transforms.t_camera_lidar, radar[valid_radar], transforms.t_camera_radar, image, transforms.camera_projection_matrix,
            #                    title=f'{label["label_class"]}')

            bboxes.append({'label_class': label['label_class'],
                                                 'track_id': label['track_id'],
                                                 'dimensions': np.array([label['l'], label['w'], label['h']]), 
                                                 'corners_3d': new_corners_3d, 
                                                 'center_3d': center[:3],  
                                                 'valid_lidar': valid_lidar,
                                                 'valid_radar': valid_radar,
                                                 'score': label['score']})
        return bboxes

    def compute_velocity(self, bboxes1, bboxes2, transforms1, transforms2, timestep):
        # pos1_ego = homogeneous_transformation(np.array([[0,0,0,1]]), transforms1.t_odom_camera)
        # pos2_ego = homogeneous_transformation(np.array([[0,0,0,1]]), transforms2.t_odom_camera)
        # print(f"compute velocity: {pos1_ego}, {pos2_ego}, {np.linalg.norm(pos2_ego - pos1_ego)}")

        for i, bbox1 in enumerate(bboxes1):
            id1 = bbox1['track_id']
            # if id1==1516: pass
            # else: continue
            pos1 = bbox1['center_3d']
            pos1 = homogeneous_coordinates(pos1[None, :])
            pos1_odom = homogeneous_transformation(pos1, transforms1.t_odom_camera @ transforms1.t_camera_lidar)

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
                pos2 = homogeneous_coordinates(pos2[None, :])
                pos2_odom = homogeneous_transformation(pos2, transforms2.t_odom_camera @ transforms2.t_camera_lidar)

                velocity = (pos2_odom - pos1_odom)[0, 0:3] / timestep 
                bbox1['velocity'] = velocity
                # print(f"compute velocity: {bbox1['velocity']}, {np.linalg.norm(velocity)}, {np.linalg.norm(pos2_odom[0, 0:3] - pos1_odom[0, 0:3]) / timestep}, {timestep}")
            else:
                # print('No track id!')
                bbox1['velocity'] = np.zeros((3))-1e8
        return bboxes1

    ############################################################################################
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

    def plot_radar_pcl(self, max_distance_threshold, min_distance_threshold):
        uvs, points_depth = project_pcl_to_image(point_cloud=self.frame_data_loader.radar_data,
                                                 t_camera_pcl=self.frame_transformations.t_camera_radar,
                                                 camera_projection_matrix=self.frame_transformations.camera_projection_matrix,
                                                 image_shape=self.frame_data_loader.image.shape)

        min_max_idx = min_max_filter(points=points_depth,
                                     max_value=max_distance_threshold,
                                     min_value=min_distance_threshold)
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]

        plt.scatter(uvs[:, 0], uvs[:, 1], c=-points_depth, alpha=0.8, s=(70 / points_depth) ** 2, cmap='jet')

    def plot_radar_image(self, points, transformation, image, projection_matrix, max_distance_threshold=50.0, min_distance_threshold=0.0):
        uvs, points_depth = project_pcl_to_image(point_cloud=points,
                                                 t_camera_pcl=transformation,
                                                 camera_projection_matrix=projection_matrix,
                                                 image_shape=image.shape)

        min_max_idx = min_max_filter(points=points_depth,
                                     max_value=max_distance_threshold,
                                     min_value=min_distance_threshold)
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]

        plt.scatter(uvs[:, 0], uvs[:, 1], c=-points_depth, alpha=0.8, s=(70 / points_depth) ** 2, cmap='jet')

    def plot_lidar_image(self, points, transformation, image, projection_matrix, max_distance_threshold=50.0, min_distance_threshold=0.0):
        uvs, points_depth = project_pcl_to_image(point_cloud=points,
                                                 t_camera_pcl=transformation,
                                                 camera_projection_matrix=projection_matrix,
                                                 image_shape=image.shape)

        min_max_idx = min_max_filter(points=points_depth,
                                     max_value=max_distance_threshold,
                                     min_value=min_distance_threshold)
        uvs = uvs[min_max_idx]
        points_depth = points_depth[min_max_idx]

        plt.scatter(uvs[:, 0], uvs[:, 1], c=-points_depth, alpha=0.4, s=1, cmap='jet')

    def visualize_all(self, lidar, transformation_lidar, radar, transformation_radar, image, projection_matrix, title=None):
        fig = plt.figure(figsize=(12, 8))
        fig.set_dpi(150)
        plt.clf()
        self.plot_lidar_image(lidar, transformation_lidar, image,  projection_matrix)
        self.plot_radar_image(radar, transformation_radar, image,  projection_matrix)
        plt.imshow(image, alpha=1)
        plt.axis('off')
        if title is not None: plt.title(title)
        plt.show()
        plt.close(fig)

    ########################################################################################
    # # # Pytorch dataloader
    def __len__(self):
        # print('total samples: ', len(self.frame_list))
        return len(self.frame_list)

    def __getitem__(self, idx):
        try:
            idx1 = idx
            idx2 = idx1+self.timestep if idx1+self.timestep<self.__len__() else idx1-self.timestep
            # print('idx: ', idx, idx1, idx2, self.frame_list[idx1], self.frame_list[idx2])
            timestep = np.abs(int(self.frame_list[idx1]) - int(self.frame_list[idx2])) * 0.1
            image1, lidar1, radar1, transforms1, bboxes1 = self.load_frame(idx1)
            image2, lidar2, radar2, transforms2, bboxes2 = self.load_frame(idx2)
            bboxes1 = self.compute_velocity(bboxes1, bboxes2, transforms1, transforms2, timestep)

            ############################################################################
            # # # debug
            # # # check ego motion
            # print(f"t_odom_camera: {transforms1.t_odom_camera} {transforms2.t_odom_camera}")
            # lidar1_homo = homogeneous_coordinates(lidar1[:, 0:3])
            # lidar2_homo = homogeneous_coordinates(lidar2[:, 0:3])
            # lidar1_odom = homogeneous_transformation(lidar1_homo, transforms1.t_odom_camera @ transforms1.t_camera_lidar)
            # lidar2_odom = homogeneous_transformation(lidar2_homo, transforms2.t_odom_camera @ transforms2.t_camera_lidar)
            # self.visualize_pcd(np.concatenate([lidar1_odom, lidar2_odom], axis=0)[:, 0:3],  np.hstack([np.zeros((len(lidar1))), np.ones((len(lidar2)))]), title=f'ego motion')

            # for bbox1 in bboxes1:
            #     # # if bbox1['label_class'] == 'Pedestrian': 
            #     # if bbox1['track_id'] == 1516: 
            #     #     pass
            #     # else:
            #     #     continue
            #     valid_lidar = bbox1['valid_lidar']
            #     valid_radar = bbox1['valid_radar']
            #     velocity = np.linalg.norm(bbox1['velocity'][0:2])
            #     # self.visualize_pcd(lidar1[:, 0:3], valid_lidar, title=f'lidar_{sum(valid_lidar)}')
            #     # self.visualize_pcd(radar1[:, 0:3], valid_radar, title=f'radar_{sum(valid_radar)}')
            #     self.visualize_all(lidar1[valid_lidar], transforms1.t_camera_lidar, radar1[valid_radar], transforms1.t_camera_radar, image1, transforms1.camera_projection_matrix,
            #                    title=f"{idx} {self.frame_list[idx]}, id: {bbox1['track_id']}, {bbox1['label_class']}, lidar points: {sum(valid_lidar)}, radar points: {sum(valid_radar)}, velocity: {velocity} m/s")

            ############################################################################
            # # # array of center_3d and velocity 
            cls_labels = []
            dists = []
            velos = []
            nums_lidar = []
            nums_radar = []
            areas = []        
            for bbox1 in bboxes1:
                cls_labels.append(cls_to_idx[bbox1['label_class'].lower()])
                dists.append(np.linalg.norm(bbox1['center_3d'][0:2]))
                velos.append(np.linalg.norm(bbox1['velocity'][0:2]))
                nums_lidar.append(sum(bbox1['valid_lidar']))
                nums_radar.append(sum(bbox1['valid_radar']))
                areas.append(np.prod(bbox1['dimensions'][0:2]))

            return image1, lidar1, radar1, np.array(cls_labels), np.array(dists), np.array(velos), np.array(nums_lidar), np.array(nums_radar), np.array(areas)
        except:
            # Not all poses are available in VoD
            return None, None, None, None, None, None, None, None, None

if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="Generate the statistics summary of the nuscenes dataset.")

    parser.add_argument(
        "--dataset_path",
        "-i",
        type=Path,
        help="Give a path to the dataset.",
        default='/media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset/VOD/view_of_delft_PUBLIC',
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
    dataset = VOD_dataset(root_dir=args.dataset_path, timestep=1)
    # https://discuss.pytorch.org/t/is-a-dataset-copied-as-part-of-dataloader-with-multiple-workers/112924
    def collate(batch):
        # print('collate batch: ', len(batch), len(batch[0]))
        return batch

    kwargs = {
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 8, 
        "pin_memory": False,
        "collate_fn": collate,
    }
    data_loader = torch.utils.data.DataLoader(
        dataset,
        **kwargs,
    )

    # ####################################################################################################################################
    # # ### HISTOGRAM
    distances = np.arange(0, 200+1e-8, 1.0)
    velocities = np.arange(0, 50+1e-8, 0.2)

    hist_dist = np.zeros((len(distances)-1, 2))
    hist_velo = np.zeros((len(velocities)-1, 2))
    hist_dist[:, 0] = distances[0:-1]
    hist_velo[:, 0] = velocities[0:-1]

    for i, batch in enumerate(tqdm(data_loader)):
        img, lidar, radar, labels, dists, velos, nums_lidar, nums_radar, areas = batch[0]
        if img is None: continue

        bin_dist, edge_dist = np.histogram(dists, distances)
        # print('edge_dist: ', edge_dist.shape, edge_dist)
        hist_dist[:, 1] += bin_dist

        bin_velo, edge_velo = np.histogram(velos, velocities)
        # print('edge_velo: ', edge_velo, edge_velo.max(), edge_velo.min())
        hist_velo[:, 1] += bin_velo

    np.savez(args.output.joinpath("distance_velocity_vod.npz"), hist_dist = hist_dist, hist_velo = hist_velo, dist_bins=distances, velo_bins=velocities)    

    ###################################################################################################################################
    ### POINT CLOUD ANALYSIS
    distances = np.arange(0, 200+1e-8, 50)
    distance_bins = list(zip(distances[:-1], distances[1:]))
    # print('distances: ', distances, distance_bins)

    structured_summary = {"name": "vod"}
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
    # print('len(dataloader): ', len(data_loader))
    for i, batch in enumerate(tqdm(data_loader)):
        assert len(batch)==1
        img, lidar, radar, labels, dists, velos, nums_lidar, nums_radar, areas = batch[0]
        structured_summary[SummaryKeys.TOTAL_NO_FRAMES.full_name] += len(batch)
        # print('var size: ', img.shape, lidar.shape, radar.shape, labels.shape, dists.shape, velos.shape, nums_lidar.shape, areas.shape)
        if img is None: continue
        for label, dist, velo, num_lidar, num_radar, area in zip(labels, dists, velos, nums_lidar, nums_radar, areas):
            if int(label)<0:  continue  # -1 indicates an instance belongs to none of the four categories.
            # print(f'summary per class: {i}, {bosch_cls[int(class_name)]}')
            label, dist, velo, num_lidar, num_radar, area = int(label), float(dist), float(velo), int(num_lidar), int(num_radar), float(area)
            class_name = bosch_cls[label]

            distance_bin_str = get_distance_str(dist, distance_bins)
            if distance_bin_str is None:
                continue

            structured_summary["box_results"][class_name][distance_bin_str][
                SummaryKeys.TOTAL_POINTS_RADAR.full_name
            ] += num_radar
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
        bins_lidar, edges_dist = np.histogram(np.linalg.norm(lidar[:, 0:2], axis=1), distances) # bev distance
        bins_radar, _ = np.histogram(np.linalg.norm(radar[:, 0:2], axis=1), distances) # bev distance
        # print('edges_dist: ', bins_lidar.shape, edges_dist.shape, edges_dist)

        for j, edge_dist in enumerate(edges_dist[0:-1]): 
            # print(f'summary per frame, total points: {j}, {edge_dist}')
            distance_bin_str = get_distance_str(edge_dist, distance_bins)
            structured_summary["frame_results"][distance_bin_str][
                SummaryKeys.TOTAL_POINTS_LIDAR.full_name
            ] += int(bins_lidar[j])
            structured_summary["frame_results"][distance_bin_str][
                SummaryKeys.TOTAL_POINTS_RADAR.full_name
            ] += int(bins_radar[j])

    structured_summary = perform_final_calculations(structured_summary)
    structured_summary = sort_summary(structured_summary)

    # save summary_vod to json
    with open(args.output.joinpath("summary_vod.json"), "w", encoding="utf-8") as handle:
        json.dump(structured_summary, handle, indent=2)

    print(f"Saved summary_nuscenes.vod to {args.output.absolute()}")
