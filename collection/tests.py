from django.test import TestCase


class CollectionTest(TestCase):
	# check that the different pages appear
	def test_landing_page(self):
		r = self.client.get('/')
		self.assertEqual(r.status_code, 200)

	def test_about_page(self):
		r = self.client.get('/about/')
		self.assertEqual(r.status_code, 200)

	def test_subscribe_page(self):
		r = self.client.get('/subscribe/')
		self.assertEqual(r.status_code, 200)

	def test_contact_page(self):
		r = self.client.get('/contact/')
		self.assertEqual(r.status_code, 200)

