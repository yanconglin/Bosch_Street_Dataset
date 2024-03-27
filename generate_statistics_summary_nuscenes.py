"""
Generate the statistics summary of the nuscenes dataset and export it as json.

This json is needed for the plot_statistics.py script.
"""
# Copyright 2024 Robert Bosch GmbH and its subsidiaries
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from plot_statistics import SummaryKeys
from rich.progress import track


def summarize_data_from_nuscenes(dataset_path: Path, distance_limits, versions=["v1.0-trainval", "v1.0-test"]):
    """
    Collect the relevant data from the nuscenes dataset and summarize it.

    :param dataset_path: The path to the dataset.
    :param version: The version of the dataset.

    :return: The collected box data.
    """
    structured_summary = {"name": "nuscenes"}
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

    distance_bins = list(zip(distance_limits[:-1], distance_limits[1:]))

    for version in versions:
        nusc = NuScenes(version=version, dataroot=str(dataset_path), verbose=True)

        frames = track(
            nusc.sample,
            description=f"Going through each nuscenes sample frame at {dataset_path.joinpath(version).absolute()}.",
            transient=True,
        )

        for frame in frames:
            # get bounding box
            sensor = "LIDAR_TOP"
            lidar_top_data = nusc.get("sample_data", frame["data"][sensor])
            ego_pose = nusc.get("ego_pose", lidar_top_data["ego_pose_token"])
            structured_summary[SummaryKeys.TOTAL_NO_FRAMES.full_name] += 1

            # got through all and collect relevant data for each box
            for annotation_token in frame["anns"]:
                annotation = nusc.get("sample_annotation", annotation_token)
                class_name = map_nuscenes_class_to_devkit_class(annotation["category_name"])
                # Calculate distance to ego vehicle
                diff_xy = annotation["translation"][:2] - np.array(ego_pose["translation"][:2])
                distance = np.linalg.norm(diff_xy)

                # Only add nonempty boxes
                if (annotation["num_lidar_pts"] > 0) or (annotation["num_radar_pts"] > 0):
                    area = annotation["size"][0] * annotation["size"][1]

                    distance_bin_str = get_distance_str(distance, distance_bins)
                    if distance_bin_str is None:
                        continue

                    structured_summary["box_results"][class_name][distance_bin_str][
                        SummaryKeys.TOTAL_POINTS_RADAR.full_name
                    ] += annotation["num_radar_pts"]
                    structured_summary["box_results"][class_name][distance_bin_str][
                        SummaryKeys.TOTAL_POINTS_LIDAR.full_name
                    ] += annotation["num_lidar_pts"]
                    structured_summary["box_results"][class_name][distance_bin_str][
                        SummaryKeys.TOTAL_AREA.full_name
                    ] += area
                    structured_summary["box_results"][class_name][distance_bin_str][
                        SummaryKeys.TOTAL_NO_BOXES.full_name
                    ] += 1

            # Get number of radar points in current frame
            for sensor_name in frame["data"].keys():
                if "RADAR" in sensor_name:
                    sensor = nusc.get("sample_data", frame["data"][sensor_name])
                    point_cloud = RadarPointCloud.from_file(str(dataset_path.joinpath(Path(sensor["filename"]))))
                elif "LIDAR" in sensor_name:
                    sensor = nusc.get("sample_data", frame["data"][sensor_name])
                    point_cloud = LidarPointCloud.from_file(str(dataset_path.joinpath(Path(sensor["filename"]))))
                else:
                    continue
                cal_sensor = nusc.get("calibrated_sensor", sensor["calibrated_sensor_token"])
                point_cloud.translate(
                    np.array(cal_sensor["translation"])
                )  # remark: Since we need radial distance, we don't need to rotate the point cloud
                radial_dist = np.linalg.norm(point_cloud.points[:2, :], axis=0)
                for dist in radial_dist:
                    distance_bin_str = get_distance_str(dist, distance_bins)
                    if distance_bin_str is not None:
                        if "RADAR" in sensor_name:
                            structured_summary["frame_results"][distance_bin_str][
                                SummaryKeys.TOTAL_POINTS_RADAR.full_name
                            ] += 1
                        elif "LIDAR" in sensor_name:
                            structured_summary["frame_results"][distance_bin_str][
                                SummaryKeys.TOTAL_POINTS_LIDAR.full_name
                            ] += 1

    return structured_summary


def perform_final_calculations(structured_summary):
    """Perform final calculations for the plots on the structured summary.

    :param structured_summary: The structured summary to perform the calculations on.
    :return: The structured summary with the final calculations.
    """
    for class_name, results in structured_summary["box_results"].items():
        for distance_bin, values in results.items():
            structured_summary["box_results"][class_name][distance_bin][SummaryKeys.DENSITY_RADAR.full_name] = (
                values[SummaryKeys.TOTAL_POINTS_RADAR.full_name] / values[SummaryKeys.TOTAL_AREA.full_name]
            )
            structured_summary["box_results"][class_name][distance_bin][SummaryKeys.DENSITY_LIDAR.full_name] = (
                values[SummaryKeys.TOTAL_POINTS_LIDAR.full_name] / values[SummaryKeys.TOTAL_AREA.full_name]
            )
            structured_summary["box_results"][class_name][distance_bin][SummaryKeys.DENSITY_PER_BOX_RADAR.full_name] = (
                values[SummaryKeys.TOTAL_POINTS_RADAR.full_name] / values[SummaryKeys.TOTAL_NO_BOXES.full_name]
            )
            structured_summary["box_results"][class_name][distance_bin][SummaryKeys.DENSITY_PER_BOX_LIDAR.full_name] = (
                values[SummaryKeys.TOTAL_POINTS_LIDAR.full_name] / values[SummaryKeys.TOTAL_NO_BOXES.full_name]
            )
    for distance_bin in structured_summary["frame_results"].keys():
        structured_summary["frame_results"][distance_bin][SummaryKeys.AVERAGE_POINTS_PER_FRAME_RADAR.full_name] = (
            structured_summary["frame_results"][distance_bin][SummaryKeys.TOTAL_POINTS_RADAR.full_name]
            / structured_summary[SummaryKeys.TOTAL_NO_FRAMES.full_name]
        )
        structured_summary["frame_results"][distance_bin][SummaryKeys.AVERAGE_POINTS_PER_FRAME_LIDAR.full_name] = (
            structured_summary["frame_results"][distance_bin][SummaryKeys.TOTAL_POINTS_LIDAR.full_name]
            / structured_summary[SummaryKeys.TOTAL_NO_FRAMES.full_name]
        )
    return structured_summary


def get_distance_str(distance, distance_bins):
    """Get the distance bin string for the given distance.

    :param distance: The distance to get the bin for.
    :param distance_bins: The distance bins to use.
    :return: The distance bin string.
    """
    for distance_bin in distance_bins:
        if distance_bin[0] <= distance < distance_bin[1]:
            distance_bin_str = f"{distance_bin[0]}m to {distance_bin[1]}m"
            return distance_bin_str
    return None


def sort_summary(summary):
    """
    Sort the summary dictionary.

    :param summary: The summary to sort.
    :return: The sorted summary.
    """
    # Sort after distance bins
    for class_name, results in summary["box_results"].items():
        summary["box_results"][class_name] = dict(sorted(results.items()))
    # Sort after class names
    summary["box_results"] = dict(sorted(summary["box_results"].items()))

    return summary


def map_nuscenes_class_to_devkit_class(nuscenes_class):
    """
    Maps the nuscenes class to the devkit class.

    :param nuscenes_class: The nuscenes class name.
    :return: The devkit class name.
    """
    devkit_class = "NotRelevant"
    if "human" in nuscenes_class:
        devkit_class = "VulnerableRoadUser"
    elif "motorcycle" in nuscenes_class:
        devkit_class = "RidableVehicle"
    elif "car" in nuscenes_class:
        devkit_class = "PassengerCar"
    elif "bus" in nuscenes_class:
        devkit_class = "LargeVehicle"

    return devkit_class


def statistics_summary_main(dataset_path: Path, output_dir: Path):
    """
    Main function to create the statistics summary of nuscenes dataset.

    :param dataset_path: The path to the nuscenes dataset.
    :param output_dir: The path to store the output json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    distances = [0, 20, 40, 60, 80, 200]
    # versions = ["v1.0-trainval", "v1.0-test"] # Full dataset
    versions = ["v1.0-mini"]  # For debugging

    structured_summary = summarize_data_from_nuscenes(dataset_path, distances, versions=versions)
    structured_summary = perform_final_calculations(structured_summary)
    structured_summary = sort_summary(structured_summary)

    # save summary_nuscenes to json
    with open(output_dir.joinpath("summary_nuscenes.json"), "w", encoding="utf-8") as handle:
        json.dump(structured_summary, handle, indent=2)

    print(f"Saved summary_nuscenes.json to {output_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the statistics summary of the nuscenes dataset.")

    parser.add_argument(
        "--dataset_path",
        "-i",
        type=Path,
        required=True,
        help="Give a path to the dataset.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=False,
        help="Give a path to store the output json.",
    )

    args = parser.parse_args()
    statistics_summary_main(Path(args.dataset_path), Path(args.output))
