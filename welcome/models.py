from django.db import models

class PageView(models.Model):
	hostname = models.CharField(max_length=32)
	timestamp = models.DateTimeField(auto_now_add=True)
	id = models.BigAutoField(primary_key=True)