go run test-go/main.go ~/Dataset/VeReMi-Dataset/DoSRandomSybil_0709/VeReMi_28800_32400_2022-9-13_21.8.11/VeReMi_28800_32400_2022-9-13_21\:8\:11/ 28800 32400
cp output.csv dosrandomsybil.csv
python -m code.data.build_gnn_dataset --attack_intensity_list 1.0
python -m code.train.gnn_train --exp_name single_dosrandomsybil_gnn --use_timestamp 0

