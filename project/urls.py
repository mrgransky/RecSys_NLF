from django.conf import settings
from django.contrib import admin
from django.urls import path, include

from welcome.views import index, health
from recsys_app.views import main_page

urlpatterns = [
	path('admin/', admin.site.urls),
	############# only welcome templates #############
	# path('', index, name='index'),
	# path('', health, name='health'),
	############# only welcome templates #############
	path('', main_page, name='index'),
]

if settings.DEBUG:
	import debug_toolbar
	urlpatterns = [
		path('__debug__/', include(debug_toolbar.urls)),
	] + urlpatterns
