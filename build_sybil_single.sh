go run test-go/main.go ~/Dataset/VeReMi-Dataset/GridSybil_0709/VeReMi_28800_32400_2025-11-15_13.57.9/VeReMi_28800_32400_2025-11-15_13\:57\:9/ 28800 32400
cp output.csv sybil.csv
python -m code.data.build_gnn_dataset --attack_intensity_list 1.0 
python -m code.train.gnn_train --exp_name single_sybil_gnn --use_timestamp 0
