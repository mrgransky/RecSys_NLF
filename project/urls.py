from django.conf import settings
from django.urls import include, path
from django.contrib import admin

from welcome.views import index, health
from recsys_app.views import main_page, about_us_page, check_password, instruction_page, process_feedback, track_click

urlpatterns = [
	path('admin/', admin.site.urls),
	# ############# only welcome templates #############
	path('health/', health, name="health"),
	# path('', index, name='home'),
	# ############# only welcome templates #############
	path('', check_password, name='check_password'),
	# path('', main_page, name='index'),
	path('home/', main_page, name='main_page'),
	path('home/query=<str:query>', main_page, name='main_page_with_query'),
	path('about_us/', about_us_page, name='about_us'),
	path('help/', instruction_page, name='instruction'),
	path('process_feedback/', process_feedback, name='process_feedback'),
	path('track_click/', track_click, name='track_click'),
]

if settings.DEBUG:
	import debug_toolbar
	urlpatterns = [
		path('__debug__/', include(debug_toolbar.urls)),
	] + urlpatterns