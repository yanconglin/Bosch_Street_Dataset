"""
Generate the statistics plots from the dataset summary jsons.
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
from datetime import datetime
from enum import Enum
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

dataset_colors = ["blue", "green", "red", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]


class SummaryKeys(Enum):
    """Enum for the names in the summary."""

    DENSITY_RADAR = 0, "density_radar"
    DENSITY_PER_BOX_RADAR = 1, "density_per_box_radar"
    DENSITY_LIDAR = 2, "density_lidar"
    DENSITY_PER_BOX_LIDAR = 3, "density_per_box_lidar"
    TOTAL_POINTS_RADAR = 4, "total_points_radar"
    TOTAL_POINTS_LIDAR = 5, "total_points_lidar"
    TOTAL_AREA = 6, "total_area"
    TOTAL_NO_BOXES = 7, "total_no_boxes"
    TOTAL_NO_FRAMES = 8, "total_no_frames"
    AVERAGE_POINTS_PER_FRAME_RADAR = 9, "average_radar_points_per_frame"
    AVERAGE_POINTS_PER_FRAME_LIDAR = 10, "average_lidar_points_per_frame"

    def __init__(self, index_pos: int, full_name: str):
        self.full_name = full_name
        self.index_pos = index_pos


def draw_point_cloud_density_plots(dataset_summaries: dict, output_dir, modality="Radar"):
    """Draw the point cloud density plots.

    :param dataset_summaries: Dictionary containing the dataset summaries.
    :param output_dir: Output directory to save the plots.
    :param modality: Modality to plot the density for. Either "Radar" or "Lidar".
    """
    # Settings
    width = 1200
    height = 800
    vertical_spacing = 0.07

    subplot_font_size = 13
    considered_classes = [
        "PassengerCar",
        "VulnerableRoadUser",
        "RidableVehicle",
        "LargeVehicle",
    ]

    figure_per_area = make_subplots(
        rows=len(considered_classes),
        cols=1,
        # shared_xaxes=True,
        y_title="Density / (points/m²)",
        x_title="Radial distance /m",
        vertical_spacing=vertical_spacing,
        subplot_titles=considered_classes,
    )
    figure_per_box = make_subplots(
        rows=len(considered_classes),
        cols=1,
        # shared_xaxes=True,
        y_title="Density / (points/object)",
        x_title="Radial distance /m",
        vertical_spacing=vertical_spacing,
        subplot_titles=considered_classes,
    )

    for dataset_idx, (dataset_name, dataset_summary) in enumerate(dataset_summaries.items()):
        color = dataset_colors[dataset_idx % len(dataset_colors)]

        for class_idx, (class_name, class_summary) in enumerate(dataset_summary["box_results"].items()):
            if class_name not in considered_classes:
                continue

            if class_idx == 0:
                show_legend = True
            else:
                show_legend = False

            if modality == "Radar":
                y_per_area = [value[SummaryKeys.DENSITY_RADAR.full_name] for value in class_summary.values()]
                y_per_box = [value[SummaryKeys.DENSITY_PER_BOX_RADAR.full_name] for value in class_summary.values()]
            else:
                y_per_area = [value[SummaryKeys.DENSITY_LIDAR.full_name] for value in class_summary.values()]
                y_per_box = [value[SummaryKeys.DENSITY_PER_BOX_LIDAR.full_name] for value in class_summary.values()]

            x, y_per_area = sort_list_for_display(list(class_summary.keys()), y_per_area)
            x, y_per_box = sort_list_for_display(list(class_summary.keys()), y_per_box)

            figure_per_area.add_trace(
                go.Bar(
                    name=f"{dataset_name}",
                    x=x,
                    y=y_per_area,
                    marker={"color": color},
                    showlegend=show_legend,
                ),
                row=considered_classes.index(class_name) + 1,
                col=1,
            )

            figure_per_box.add_trace(
                go.Bar(name=f"{dataset_name}", x=x, y=y_per_box, marker={"color": color}, showlegend=show_legend),
                row=considered_classes.index(class_name) + 1,
                col=1,
            )

    figure_per_area.update_annotations(font_size=subplot_font_size)
    figure_per_box.update_annotations(font_size=subplot_font_size)
    figure_per_area.update_layout(
        title=f"Point Cloud Density Comparison - Points per Area ({modality})",
        height=height,
        width=width,
    )
    figure_per_box.update_layout(
        title=f"Point Cloud Density Comparison - Points per Box ({modality})",
        height=height,
        width=width,
    )

    figure_per_area.write_image(output_dir.joinpath(f"density_comparison_points_per_area_{modality}.png"))
    figure_per_box.write_image(output_dir.joinpath(f"density_comparison_points_per_box_{modality}.png"))

    figure_per_area.show()
    figure_per_box.show()

    print(f"Saved density plots to {output_dir}")


def draw_point_points_per_frame_plots(dataset_summaries: dict, output_dir, modality="Radar"):
    """Draw the point points per frame plots.
    :param dataset_summaries: Dictionary containing the dataset summaries.
    :param output_dir: Output directory to save the plots.
    :param modality: Modality to plot the density for. Either "Radar" or "Lidar".
    """

    # Settings
    width = 1200
    height = 800

    figure = go.Figure()
    for dataset_idx, (dataset_name, dataset_summary) in enumerate(dataset_summaries.items()):
        if "frame_results" not in dataset_summary:
            continue

        color = dataset_colors[dataset_idx % len(dataset_colors)]
        if modality == "Radar":
            y = [
                distance_data[SummaryKeys.AVERAGE_POINTS_PER_FRAME_RADAR.full_name]
                for distance_data in dataset_summary["frame_results"].values()
            ]
        else:
            y = [
                distance_data[SummaryKeys.AVERAGE_POINTS_PER_FRAME_LIDAR.full_name]
                for distance_data in dataset_summary["frame_results"].values()
            ]
        x = list(dataset_summary["frame_results"].keys())
        x = [x_ for x_ in x if x_ !='null'] # argoverse-v2: some classes are not in the official class list!
        x, y = sort_list_for_display(x, y)
        figure.add_trace(go.Bar(name=f"{dataset_name}", x=x, y=y, marker={"color": color}))

    figure.update_layout(
        title=f"Average Points per Frame ({modality})",
        xaxis_title="Radial distance /m",
        yaxis_title="Average Points per Frame",
        height=height,
        width=width,
    )

    figure.write_image(output_dir.joinpath(f"average_points_per_frame_{modality}.png"))
    figure.show()


def plot_dataset_statistics_main(input_dir: Path, output_dir: Path):
    """Main function to run the dataset statistics.

    :param config_path: Path to the evaluation configuration.
    :param output_dir: Output directory to save the plots.
    """
    script_start_time = datetime.now()

    # Add start time to output dir
    output_dir = output_dir.joinpath(script_start_time.strftime("%Y%m%d_%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all json files in input_dir
    json_files = list(input_dir.glob("*.json"))
    json_files.sort()
    dataset_summaries = {}
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as handle:
            dataset_summary = json.load(handle)
            dataset_summaries[dataset_summary["name"]] = dataset_summary

    # Draw the plots
    draw_point_points_per_frame_plots(dataset_summaries, output_dir=output_dir, modality="Radar")
    draw_point_points_per_frame_plots(dataset_summaries, output_dir=output_dir, modality="Lidar")
    draw_point_cloud_density_plots(dataset_summaries, output_dir=output_dir, modality="Radar")
    draw_point_cloud_density_plots(dataset_summaries, output_dir=output_dir, modality="Lidar")

def sort_list_for_display(x, y):
    idx = sorted(range(len(x)), key=lambda k: float(x[k][0:3]))
    sorted_x = [x[i] for i in idx]
    sorted_y = [y[i] for i in idx]
    return sorted_x, sorted_y 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dataset statistics.")

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Give a path to the directory containing all the summary json "
        + "files of the datasets that should be included in the plots.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=False,
        help="Give a path to store the output.",
    )

    args = parser.parse_args()
    plot_dataset_statistics_main(args.input, args.output)
