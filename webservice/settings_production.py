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
import dj_database_url

# Everything below will override our standard settings:

# ----- DATABASES -----
if 'RDS_HOSTNAME' in os.environ:
    # Your EB RDS is POSTGRES, not MySQL
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.environ['RDS_DB_NAME'],
            'USER': os.environ['RDS_USERNAME'],
            'PASSWORD': os.environ['RDS_PASSWORD'],
            'HOST': os.environ['RDS_HOSTNAME'],
            'PORT': os.environ.get('RDS_PORT', '5432'),
        }
    }
elif os.getenv('DATABASE_URL'):
    # fallback: parse DATABASE_URL if you ever set it
    DATABASES['default'] = dj_database_url.config()
else:
    # last resort: local sqlite so the app can still boot
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
        }
    }

DB_CONNECTION_URL = DB_CONNECTION_URL = os.getenv('DATABASE_URL', 'awseb-e-jivwhga8ui-stack-awsebrdsdatabase-whgz4rmetb10.cee9izytbdnd.eu-central-1.rds.amazonaws.com')


SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
USE_X_FORWARDED_HOST = True
ALLOWED_HOSTS  = ['*', 'www.treeofknowledge.ai', 'treeofknowledge.ai', 'Treeofknowledge-production-5.eba-ffmsq3fy.eu-central-1.elasticbeanstalk.com']
DEBUG = False

# ----- middleware: prepend whitenoise, don't replace -----
MIDDLEWARE = ['whitenoise.middleware.WhiteNoiseMiddleware'] + MIDDLEWARE  # use the list from base settings


# ----- static -----
STATIC_URL = '/static/'
STATIC_ROOT = '/var/app/current/static/'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'
# STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'  # <-this is super strict: it will make 'eb deploy' fail if there is a reference to a static file that no longer is in the right folder



