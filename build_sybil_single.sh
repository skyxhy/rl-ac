go run test-go/main.go ~/Dataset/VeReMi-Dataset/GridSybil_0709/VeReMi_28800_32400_2025-11-15_13.57.9/VeReMi_28800_32400_2025-11-15_13\:57\:9/ 28800 32400
cp output.csv sybil.csv
python -m rlac.data.datasets --attack_intensity_list 1.0 
python -m rlac.run.train_gnn --exp_name single_sybil_gnn --use_timestamp 0
