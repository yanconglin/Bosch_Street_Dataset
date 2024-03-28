# python generate_statistics_summary_vod.py -i /media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset/VOD/view_of_delft_PUBLIC -o ./vod
python plot_histogram.py -i  ./vod/distance_velocity_vod.npz -o ./vod
python plot_statistics.py -i  ./vod -o ./vod


# # NCCL_P2P_DISABLE=1 python generate_statistics_summary_av2.py -i /media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset -o ./av2
# # OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python generate_statistics_summary_av2.py -i /media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset -o ./av2
# python generate_statistics_summary_av2.py -i /media/yanconglin/4408c7fc-2531-4bdd-9dfd-421b2cc2246e/Dataset -o ./av2
# python plot_histogram.py -i  ./av2/distance_velocity_av2.npz -o ./av2
# python plot_statistics.py -i  ./av2 -o ./av2
