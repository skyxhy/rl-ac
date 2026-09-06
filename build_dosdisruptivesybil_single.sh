go run test-go/main.go  ~/Dataset/VeReMi-Dataset/DoSDisruptiveSybil_0709/VeReMi_28800_32400_2022-9-13_21.8.24/VeReMi_28800_32400_2022-9-13_21\:8\:24/ 28800 32400
cp output.csv dosdisruptivesybil.csv
python -m rlac.data.datasets --attack_intensity_list 1.0 
python -m rlac.run.train_gnn --exp_name single_dosdisruptivesybil_gnn --use_timestamp 0
