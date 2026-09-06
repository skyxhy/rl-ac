go run test-go/main.go ~/Dataset/VeReMi-Dataset/DoSRandom_0709/VeReMi_28800_32400_2022-9-12_3.0.28/VeReMi_28800_32400_2022-9-12_3\:0\:28/ 28800 32400
cp output.csv dosrandom.csv
python -m code.data.build_gnn_dataset --attack_intensity_list 1.0
python -m code.train.gnn_train --exp_name single_dosrandom_gnn --use_timestamp 0

