####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (C) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, February 2021
#####################################################################

from django import forms
from collection.models import Profile, Newsletter_subscriber, Simulation_model, Uploaded_dataset
from django.contrib.auth.models import User

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email')

class ProfileForm(forms.ModelForm):
    verbose = forms.CheckboxInput()
    class Meta:
        model = Profile
        fields = ('verbose',)


class Subscriber_registrationForm(forms.ModelForm):
	class Meta:
		model = Newsletter_subscriber
		fields = ('first_name', 'email', )


class Subscriber_preferencesForm(forms.ModelForm):
	class Meta:
		model = Newsletter_subscriber
		fields = ('is_templar', 'is_alchemist', 'is_scholar', )
		# fields = ('email', 'userid', 'first_name', 'is_templar', 'is_alchemist', 'is_scholar', 'created', 'updated',)




class UploadFileForm(forms.Form):
    file = forms.FileField()
 


class Uploaded_datasetForm2(forms.ModelForm):
	data_generation_date = forms.DateField(required=False)
	class Meta:
		model = Uploaded_dataset
		fields = ('data_source', 'data_generation_date', 'correctness_of_data', )

class Uploaded_datasetForm3(forms.ModelForm):
	class Meta:
		model = Uploaded_dataset
		fields = ('object_type_id' ,'object_type_name','entire_objectInfoHTMLString', )


class Uploaded_datasetForm4(forms.ModelForm):
	class Meta:
		model = Uploaded_dataset
		fields = ('meta_data_facts', )


class Uploaded_datasetForm5(forms.ModelForm):
	data_table_json = forms.CharField()
	attribute_selection = forms.CharField()
	datetime_column = forms.CharField(required=False)
	object_identifiers = forms.CharField(required=False)
	class Meta:
		model = Uploaded_dataset
		fields = ('data_table_json', 'attribute_selection', 'datetime_column', 'object_identifiers')

class Uploaded_datasetForm6(forms.ModelForm):
	class Meta:
		model = Uploaded_dataset
		fields = ('list_of_matches', 'upload_only_matched_entities',)


class Uploaded_datasetForm7(forms.ModelForm):
	class Meta:
		model = Uploaded_dataset
		fields = ('data_generation_date', )

