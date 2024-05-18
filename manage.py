#!/usr/bin/env python
import os
import sys
##################################################
# avoid __pycache__ # DON NOT DELETE THIS LINE!!!!
sys.dont_write_bytecode = True 
##################################################


def get_project_name():
	settings_module = os.environ.get('DJANGO_SETTINGS_MODULE')
	if settings_module:
		return settings_module.split('.')[0]
	return None

def main():
	os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
	project_name = get_project_name()
	if project_name:
		print(f"Django project name: {project_name}")
	else:
		print("Unable to determine the Django project name.")		
	try:
		from django.core.management import execute_from_command_line
	except ImportError as exc:
		raise ImportError(
			"Couldn't import Django. Are you sure it's installed and "
			"available on your PYTHONPATH environment variable? Did you "
			"forget to activate a virtual environment?"
		) from exc
	execute_from_command_line(sys.argv)

# def main():
# 	os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
# 	try:
# 		from django.core.management import execute_from_command_line
# 	except ImportError as exc:
# 		raise ImportError(
# 			"Couldn't import Django. Are you sure it's installed and "
# 			"available on your PYTHONPATH environment variable? Did you "
# 			"forget to activate a virtual environment?"
# 		) from exc
# 	execute_from_command_line(sys.argv)

if __name__ == '__main__':
	main()