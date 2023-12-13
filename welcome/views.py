import os
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse

from . import database
from .models import PageView

def index(request):
	hostname = os.getenv('HOSTNAME', 'unknown')
	PageView.objects.create(hostname=hostname)

	return render(request, 'welcome/index.html', {
		'hostname': hostname,
		'database': database.info(),
		'count': PageView.objects.count()
	})

def health(request):
	"""Takes an request as a parameter and give the count of pageview objects as reponse"""
	return HttpResponse(f"Hello, Django:\ncount of pageview objects: {PageView.objects.count()}")