from django.conf import settings
from django.urls import include, path
from django.contrib import admin

from welcome.views import index, health

urlpatterns = [
	path('admin/', admin.site.urls),
	############# only welcome templates #############
	path('', index, name='index'),
	path('', health, name='health'),
	############# only welcome templates #############
]

if settings.DEBUG:
	import debug_toolbar
	urlpatterns = [
		path('__debug__/', include(debug_toolbar.urls)),
	] + urlpatterns
