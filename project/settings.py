import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /home/farid/WS_Farid/RecSys_NLF
SECRET_KEY = os.getenv(
		'DJANGO_SECRET_KEY',
		'9e4@&tw46$l31)zrqe3wi+-slqm(ruvz&se0^%9#6(_w3ui!c0'
)

DEBUG = False
# DEBUG = True # causes "Broken pipe from ('94.72.62.225', 53959)" error!
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SECURE_SSL_REDIRECT = False

ALLOWED_HOSTS = ['*']
# ALLOWED_HOSTS = ['128.214.254.157', '10.130.15.24', '127.0.0.1', 'localhost',]  # Add the IP address of your VM # $lsof -i :8000

INSTALLED_APPS = [
		'django.contrib.admin',
		'django.contrib.auth',
		'django.contrib.contenttypes',
		'django.contrib.sessions',
		'django.contrib.messages',
		'django.contrib.staticfiles',
		'debug_toolbar',
		'welcome',
		'recsys_app',
]

MIDDLEWARE = [
		'django.middleware.security.SecurityMiddleware',
		'django.contrib.sessions.middleware.SessionMiddleware',
		'django.middleware.common.CommonMiddleware',
		'django.middleware.csrf.CsrfViewMiddleware',
		'django.contrib.auth.middleware.AuthenticationMiddleware',
		'django.contrib.messages.middleware.MessageMiddleware',
		'django.middleware.clickjacking.XFrameOptionsMiddleware',
		'whitenoise.middleware.WhiteNoiseMiddleware',
		'debug_toolbar.middleware.DebugToolbarMiddleware',
]

ROOT_URLCONF = 'project.urls'

TEMPLATES = [
		{
				'BACKEND': 'django.template.backends.django.DjangoTemplates',
				'DIRS': [],
				'APP_DIRS': True,
				'OPTIONS': {
						'context_processors': [
								'django.template.context_processors.debug',
								'django.template.context_processors.request',
								'django.contrib.auth.context_processors.auth',
								'django.contrib.messages.context_processors.messages',
						],
				},
		},
]

WSGI_APPLICATION = 'project.wsgi.application'

from . import database
DATABASES = {
	'default': database.config()
}

DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'
AUTH_PASSWORD_VALIDATORS = [
	{
		'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
	},
	{
		'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
	},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True
# STATIC_URL = '/static/'
STATIC_URL = 'static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# if os.getenv('USER') == "ubuntu":
# 	STATICFILES_DIRS = (os.path.join(BASE_DIR, 'static'),) # Extra places for collectstatic to find static files
# STATICFILES_DIRS = (os.path.join(BASE_DIR, 'static'),) # Extra places for collectstatic to find static files
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')] # Extra places for collectstatic to find static files

INTERNAL_IPS = ['127.0.0.1']