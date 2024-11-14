####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (C) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, February 2021
#####################################################################

from registration.backends.simple.views import RegistrationView
from django.core.mail import EmailMultiAlternatives

class TOKRegistrationView(RegistrationView):
	def get_success_url(self, user):

		message = '''Hi ''' + user.username + ''',

Thank you for signing up to the Tree of Knowledge.'''
		email_message = EmailMultiAlternatives('Tree of Knowledge', message, 'noreply@treeofknowledge.ai', [user.email])
		email_message.send()

		return ('main_menu')