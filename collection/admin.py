from django.contrib import admin

# import the models
from collection.models import (Profile, Newsletter_subscriber, Uploaded_dataset, Data_point, Object_hierachy_tree_history, Object_types, Object, Attribute, Simulation_model, Rule, Execution_order, Likelihood_function, Rule_parameter, Logged_variable, Monte_carlo_result, Learn_parameters_result)


# automated email_hash creation for Newsletter subscribers
class Newsletter_subscriberAdmin(admin.ModelAdmin):
	model = Newsletter_subscriber
	list_display = ('email', 'userid', 'first_name','is_templar', 'is_alchemist', 'is_scholar', 'created', 'updated')
	search_fields = ('email', 'userid', 'first_name','is_templar', 'is_alchemist', 'is_scholar', 'created', 'updated')
	# prepopulated_fields = {'userid': ('email',), 'created': ('email',),'updated': ('email',),}



class Uploaded_datasetAdmin(admin.ModelAdmin):
	model = Uploaded_dataset
	list_display = ('file_name', 'file_path', 'sep', 'encoding', 'quotechar', 'escapechar', 'na_values', 'skiprows', 'header', 'created', 'updated', 'user')


class Data_pointAdmin(admin.ModelAdmin):
	model = Uploaded_dataset
	list_display = ('object_id', 'attribute_id', 'value_as_string', 'numeric_value', 'string_value', 'boolean_value', 'valid_time_start', 'valid_time_end', 'data_quality',)
	search_fields = ('object_id', 'attribute_id', 'value_as_string', 'numeric_value', 'string_value', 'boolean_value', 'valid_time_start', 'valid_time_end', 'data_quality')


# register the models
admin.site.register(Profile)
admin.site.register(Newsletter_subscriber, Newsletter_subscriberAdmin)
admin.site.register(Uploaded_dataset, Uploaded_datasetAdmin)
admin.site.register(Data_point)
admin.site.register(Object_hierachy_tree_history)
admin.site.register(Object_types)
admin.site.register(Object)
admin.site.register(Attribute)
admin.site.register(Rule)
admin.site.register(Rule_parameter)
admin.site.register(Execution_order)
admin.site.register(Simulation_model)
admin.site.register(Logged_variable)
admin.site.register(Learn_parameters_result)
admin.site.register(Monte_carlo_result)
admin.site.register(Likelihood_function)


