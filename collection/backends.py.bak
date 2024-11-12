####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (c) 2019-2040 Benedikt Kleppmann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version - see http://www.gnu.org/licenses/.
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