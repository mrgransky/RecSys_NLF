#!/bin/bash

## run using command:
## $ nohup bash run_nlf_website.sh > /dev/null 2>&1 &
## $ nohup bash run_nlf_website.sh > /media/volume/trash/NLF/nlf_website_logs.out 2>&1 & # with output saved in logs.out

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
echo "HOME DIR $HOME_DIR"
source $HOME_DIR/miniconda3/bin/activate py39

python -u manage.py runserver 0.0.0.0:8000