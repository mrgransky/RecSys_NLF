#!/bin/bash

## run using command:
## $ nohup bash nlf_website_pouta.sh > /dev/null 2>&1 &
## $ nohup bash nlf_website_pouta.sh > /media/volume/Nationalbiblioteket/trash/nlf_website_logs.out &

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error and exit immediately.
set -o pipefail # If any command in a pipeline fails, the entire pipeline will fail.

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

source $(conda info --base)/bin/activate py39
rm -rf staticfiles; python manage.py collectstatic;
python -u manage.py runserver 0.0.0.0:8000