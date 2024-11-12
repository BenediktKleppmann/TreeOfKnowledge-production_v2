####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (c) 2019-2040 Benedikt Kleppmann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version - see http://www.gnu.org/licenses/.
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

# Allow all host headers
ALLOWED_HOSTS  = ['*']

# Set debug to False
DEBUG = False

# Static asset configuration
STATICFILES_STORAGE = 'whitenoise.django.GzipManifestStaticFilesStorage'

