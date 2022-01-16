#! /bin/bash

pip install -r requirements.txt
python dataset.py
python training.py --data_dir "`pwd`/Dataset/Dataset" --yaml_file_path "`pwd`/model_config.yml"
python inference.py --model_path "`pwd`/model.h5"
