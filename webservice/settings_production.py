####################################################################
# This file is part of the Tree of Knowledge project.
#
# Copyright (c) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, November 2024
#####################################################################

# Inherit from standard settings file for default
from webservice.settings import *
import os

# Everything below will override our standard settings:

# Parse database configuration from $DATABASE_URL
import dj_database_url

if 'RDS_HOSTNAME' in os.environ:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': os.environ['RDS_DB_NAME'],
            'USER': os.environ['RDS_USERNAME'],
            'PASSWORD': os.environ['RDS_PASSWORD'],
            'HOST': os.environ['RDS_HOSTNAME'],
            'PORT': os.environ['RDS_PORT'],
        }
    }
else:
	DATABASES['default'] = dj_database_url.config()

DB_CONNECTION_URL = os.environ['DATABASE_URL']

# Honor the 'X-Forwarded-Proto' header for request.is_secure()
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_X_FORWARDED_HOST = True

# Allow all host headers
ALLOWED_HOSTS  = ['*', 'Treeofknowledge-production-5.eba-ffmsq3fy.eu-central-1.elasticbeanstalk.com']

# Set debug to False
DEBUG = False

# Static asset configuration
MIDDLEWARE = [
    'whitenoise.middleware.WhiteNoiseMiddleware',
]
#STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
STATICFILES_STORAGE = 'whitenoise.django.GzipManifestStaticFilesStorage'
STATIC_URL = '/static/'
STATIC_ROOT = '/var/app/current/static/'

