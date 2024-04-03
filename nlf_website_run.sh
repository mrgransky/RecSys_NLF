#!/bin/bash

## run using command:
## $ nohup bash nlf_website_run.sh > /dev/null 2>&1 &
## $ nohup bash nlf_website_run.sh > /media/volume/trash/NLF/nlf_website_logs.out 2>&1 & # with output saved in logs.out

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

HOME_DIR=$(echo $HOME)
echo "HOME DIR $HOME_DIR"
source $HOME_DIR/miniconda3/bin/activate py39

rm -rfv staticfiles
python manage.py collectstatic
python -u manage.py runserver 0.0.0.0:8000