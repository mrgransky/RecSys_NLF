from django.conf import settings
from django.urls import include, path
from django.contrib import admin

from welcome.views import index, health
from recsys_app.views import main_page, about_us_page, check_password

urlpatterns = [
	path('admin/', admin.site.urls),
	# ############# only welcome templates #############
	path('health/', health, name="health"),
	# path('', index, name='home'),
	# ############# only welcome templates #############
	path('', check_password, name='check_password'),
	# path('', main_page, name='index'),
	path('home/', main_page, name='main_page'),
	path('about_us/', about_us_page, name='about_us'),
]

if settings.DEBUG:
	import debug_toolbar
	urlpatterns = [
		path('__debug__/', include(debug_toolbar.urls)),
	] + urlpatterns