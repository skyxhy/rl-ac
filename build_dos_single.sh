go run test-go/main.go ~/Dataset/VeReMi-Dataset/DoS_0709/VeReMi_28800_32400_2022-9-12_3.0.15/VeReMi_28800_32400_2022-9-12_3\:0\:15/ 28800 32400
cp output.csv dos.csv
python -m code.data.build_gnn_dataset --attack_intensity_list 1.0
python -m code.train.gnn_train --exp_name single_dos_gnn --use_timestamp 0

