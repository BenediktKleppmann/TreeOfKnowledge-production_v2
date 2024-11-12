####################################################################
# This file is part of the Tree of Knowledge project.
#
# Copyright (c) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, November 2024
#####################################################################



# from registration.backends.simple.views import RegistrationView
from collection.backends import TOKRegistrationView
from registration.forms import RegistrationFormUniqueEmail
from django.contrib import admin
from collection import views
from django.views.generic import RedirectView
from django.conf.urls import include, url
from django.urls import path
from django.views.generic import TemplateView
from django.contrib.auth.views import (
    password_reset,
    password_reset_done,
    password_reset_confirm,
    password_reset_complete
)
import os

from django.conf import settings
settings.EMAIL_HOST = 'smtp.fastmail.com'
settings.EMAIL_PORT = 465
settings.EMAIL_HOST_USER = 'benedikt@kleppmann.de'
settings.DEFAULT_FROM_EMAIL = 'noreply@treeofknowledge.ai'
settings.SERVER_MAIL = 'noreply@treeofknowledge.ai'
settings.EMAIL_HOST_PASSWORD = os.environ['EMAIL_HOST_PASSWORD']
settings.EMAIL_USE_TLS = False
settings.EMAIL_USE_SSL = True
settings.EMAIL_TIMEOUT = 3600
settings.DEFAULT_CHARSET = 'utf-8'
settings.EMAIL_USE_LOCALTIME = True
settings.EMAIL_USE_LOCALTIME = True

urlpatterns = [

    # ==================================================
    # THE WEBSITE  
    # ==================================================
    url(r'^$', views.landing_page, name='landing_page'),
    url(r'^about/$', views.about, name='about'),
    url(r'^tutorial_overview/$', views.tutorial_overview, name='tutorial_overview'),
    url(r'^tutorial1/$', views.tutorial1, name='tutorial1'),
    url(r'^tutorial2/$', views.tutorial2, name='tutorial2'),
    url(r'^tutorial3/$', views.tutorial3, name='tutorial3'),
    url(r'^subscribe/$', views.subscribe, name='subscribe'),
    url(r'^contact/$', views.contact, name='contact'),
    url(r'^subscriber/(?P<userid>[-\d]+)/$', views.subscriber_page, name='subscriber_page'),

    # Registration  -------------------------------------------------------------------------
    url(r'^accounts/password/reset/$', password_reset, {'template_name': 'registration/password_reset_form.html'}, name='password_reset'),
    url(r'^accounts/password/reset/done/$', password_reset_done, {'template_name': 'registration/password_reset_done.html'}, name='password_reset_done'),
    url(r'^accounts/password/reset/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,13}-[0-9A-Za-z]{1,20})/$', password_reset_confirm, {'template_name': 'registration/password_reset_confirm.html'}, name='password_reset_confirm'),
    url(r'^accounts/password/done/$', password_reset_complete, {'template_name': 'registration/password_reset_complete.html'}, name='password_reset_complete'),
    # url(r'^accounts/register/$', RegistrationView.as_view(),name='registration_register'),
    url(r'^accounts/register/$', TOKRegistrationView.as_view(), name='registration_register'),
    url(r'^accounts/register/complete/$', TemplateView.as_view(template_name='registration/registration_complete.html'), name='registration_complete'),
    url(r'^accounts/', include('registration.backends.simple.urls')),
    url(r'^', include('django.contrib.auth.urls')),


    # ==================================================
    # THE TOOL  
    # ==================================================
    url(r'^tool/main_menu/$', views.main_menu, name='main_menu'),
    url(r'^tool/open_your_simulation/$', views.open_your_simulation, name='open_your_simulation'),
    url(r'^tool/browse_simulations/$', views.browse_simulations, name='browse_simulations'),
    url(r'^tool/profile_and_settings/$', views.profile_and_settings, name='profile_and_settings'),
    

    # Upload data  -------------------------------------------------------------------------
    url(r'^tool/upload_data1/$', views.upload_data1_new, name='upload_data1_new'),
    url(r'^tool/upload_data1/(?P<upload_id>[-\d]+)/$', views.upload_data1, name='upload_data1'),
    url(r'^tool/upload_data2/(?P<upload_id>[-\d]+)/$', views.upload_data2, name='upload_data2'),
    url(r'^tool/upload_data3/(?P<upload_id>[-\d]+)/$', views.upload_data3, name='upload_data3'),
    url(r'^tool/upload_data4/(?P<upload_id>[-\d]+)/$', views.upload_data4, name='upload_data4'),
    url(r'^tool/upload_data5/(?P<upload_id>[-\d]+)/$', views.upload_data5, name='upload_data5'),
    url(r'^tool/upload_data6A/(?P<upload_id>[-\d]+)/$', views.upload_data6A, name='upload_data6A'),
    url(r'^tool/upload_data6B/(?P<upload_id>[-\d]+)/$', views.upload_data6B, name='upload_data6B'),
    url(r'^tool/upload_data7/(?P<upload_id>[-\d]+)/$', views.upload_data7, name='upload_data7'),
    url(r'^tool/upload_data_success/(?P<number_of_datapoints_saved>[-\d]+)-(?P<new_model_id>[-\d]+)/$', views.upload_data_success, name='upload_data_success'),
    url(r'^tool/get_upload_progress/$', views.get_upload_progress, name='get_upload_progress'),

    # Helper Functions  -------------------------------------------------------------------------
    # get
    url(r'^tool/get_possible_attributes/$', views.get_possible_attributes, name='get_possible_attributes'),
    url(r'^tool/get_list_of_parent_objects/$', views.get_list_of_parent_objects, name='get_list_of_parent_objects'),
    url(r'^tool/get_list_of_objects/$', views.get_list_of_objects, name='get_list_of_objects'),
    url(r'^tool/get_attribute_details/$', views.get_attribute_details, name='get_attribute_details'),
	url(r'^tool/get_attribute_rules_old/$', views.get_attribute_rules_old, name='get_attribute_rules_old'),    
    url(r'^tool/get_object_hierachy_tree/$', views.get_object_hierachy_tree, name='get_object_hierachy_tree'),
    # url(r'^tool/get_available_variables/$', views.get_available_variables, name='get_available_variables'),
    url(r'^tool/get_object_rules/$', views.get_object_rules, name='get_object_rules'),
    url(r'^tool/get_all_pdfs/$', views.get_all_pdfs, name='get_all_pdfs'),
    url(r'^tool/get_rules_pdf/$', views.get_rules_pdf, name='get_rules_pdf'),
    url(r'^tool/get_single_pdf/$', views.get_single_pdf, name='get_single_pdf'),
    url(r'^tool/get_parameter_info/$', views.get_parameter_info, name='get_parameter_info'),
    url(r'^tool/get_simulated_parameter_numbers/$', views.get_simulated_parameter_numbers, name='get_simulated_parameter_numbers'),
    url(r'^tool/get_missing_objects_dict_attributes/$', views.get_missing_objects_dict_attributes, name='get_missing_objects_dict_attributes'),
    url(r'^tool/get_all_priors_df_and_learned_rules/$', views.get_all_priors_df_and_learned_rules, name='get_all_priors_df_and_learned_rules'),

    # check
    url(r'^tool/check_if_simulation_results_exist/$', views.check_if_simulation_results_exist, name='check_if_simulation_results_exist'),


    # complex get
    url(r'^tool/get_data_points/$', views.get_data_points, name='get_data_points'),
    url(r'^tool/get_data_from_random_object/$', views.get_data_from_random_object, name='get_data_from_random_object'),
    url(r'^tool/get_data_from_random_related_object/$', views.get_data_from_random_related_object, name='get_data_from_random_related_object'),
    url(r'^tool/get_data_from_objects_behind_the_relation/$', views.get_data_from_objects_behind_the_relation, name='get_data_from_objects_behind_the_relation'),
    url(r'^tool/get_execution_order/$', views.get_execution_order, name='get_execution_order'),

    # find
    url(r'^tool/find_suggested_attributes/$', views.find_suggested_attributes, name='find_suggested_attributes'),
    url(r'^tool/find_suggested_attributes2/$', views.find_suggested_attributes2, name='find_suggested_attributes2'),
    url(r'^tool/find_matching_entities/$', views.find_matching_entities, name='find_matching_entities'),
    url(r'^tool/find_single_entity/$', views.find_single_entity, name='find_single_entity'),    
    # save
    url(r'^tool/save_new_object_hierachy_tree/$', views.save_new_object_hierachy_tree, name='save_new_object_hierachy_tree'),
    url(r'^tool/save_new_object_type/$', views.save_new_object_type, name='save_new_object_type'),
    url(r'^tool/save_edited_object_type/$', views.save_edited_object_type, name='save_edited_object_type'),
    url(r'^tool/save_new_attribute/$', views.save_new_attribute, name='save_new_attribute'),
    url(r'^tool/save_changed_attribute/$', views.save_changed_attribute, name='save_changed_attribute'),
    url(r'^tool/save_rule/$', views.save_rule, name='save_rule'),
    url(r'^tool/save_changed_simulation/$', views.save_changed_simulation, name='save_changed_simulation'),
    # url(r'^tool/save_learned_rule/$', views.save_learned_rule, name='save_learned_rule'),
    url(r'^tool/save_changed_object_type_icon/$', views.save_changed_object_type_icon, name='save_changed_object_type_icon'),
    url(r'^tool/save_rule_parameter/$', views.save_rule_parameter, name='save_rule_parameter'),
    url(r'^tool/save_likelihood_function/$', views.save_likelihood_function, name='save_likelihood_function'),
    url(r'^tool/save_changed_execution_order/$', views.save_changed_execution_order, name='save_changed_execution_order'),
    url(r'^tool/save_new_execution_order/$', views.save_new_execution_order, name='save_new_execution_order'),
    # delete
    url(r'^tool/delete_object_type/$', views.delete_object_type, name='delete_object_type'),
    url(r'^tool/delete_attribute/$', views.delete_attribute, name='delete_attribute'),
    url(r'^tool/delete_rule/$', views.delete_rule, name='delete_rule'),
    url(r'^tool/delete_parameter/$', views.delete_parameter, name='delete_parameter'),
    url(r'^tool/delete_execution_order/$', views.delete_execution_order, name='delete_execution_order'),
    # process
    url(r'^tool/edit_column/$', views.edit_column, name='edit_column'),
    # url(r'^tool/learn_rule_from_factors/$', views.learn_rule_from_factors, name='learn_rule_from_factors'),
    # column format
    url(r'^tool/suggest_attribute_format/$', views.suggest_attribute_format, name='suggest_attribute_format'),
    url(r'^tool/get_columns_format_violations/$', views.get_columns_format_violations, name='get_columns_format_violations'),
    url(r'^tool/check_single_fact_format/$', views.check_single_fact_format, name='check_single_fact_format'),    
    
   
    # Models  -------------------------------------------------------------------------
    url(r'^tool/execution_order_scores/$', views.execution_order_scores, name='execution_order_scores'),
    url(r'^tool/get_execution_order_scores/$', views.get_execution_order_scores, name='get_execution_order_scores'),



    # Query Data  -------------------------------------------------------------------------
    url(r'^tool/query_data/$', views.query_data, name='query_data'),
    url(r'^tool/download_file1/$', views.download_file1, name='download_file1'),
    url(r'^tool/query_data/(?P<file_name>[-\d]+)-(?P<file_type>[a-z]+)/$', views.download_file2, name='download_file2'),
    

	# Simulation  -------------------------------------------------------------------------    
    url(r'^tool/edit_simulation/$', views.edit_simulation_new, name='edit_simulation_new'),
    url(r'^tool/edit_simulation/(?P<simulation_id>[-\d]+)/$', views.edit_simulation, name='edit_simulation'),
    url(r'^tool/copy_simulation/$', views.copy_simulation, name='copy_simulation'),
    url(r'^tool/get_simulation_progress/$', views.get_simulation_progress, name='get_simulation_progress'),
    url(r'^tool/analyse_simulation/(?P<simulation_id>[-\d]+)/(?P<execution_order_id>[-\d]+)/$', views.analyse_learned_parameters, name='analyse_learned_parameters'),
    url(r'^tool/analyse_simulation/(?P<simulation_id>[-\d]+)/(?P<execution_order_id>[-\d]+)/new-(?P<parameter_number>[-\d]+)/$', views.analyse_new_simulation, name='analyse_new_simulation'),
    url(r'^tool/analyse_simulation/(?P<simulation_id>[-\d]+)/(?P<execution_order_id>[-\d]+)/(?P<parameter_number>[-\d]+)/$', views.analyse_simulation, name='analyse_simulation'),
    url(r'^tool/abort_simulation/$', views.abort_simulation, name='abort_simulation'),
    url(r'^tool/run_single_monte_carlo/$', views.run_single_monte_carlo, name='run_single_monte_carlo'),


	# catching missspellt urls...
    url(r'^main_menu/$', RedirectView.as_view(pattern_name='main_menu')),
    url(r'^edit_model/$', RedirectView.as_view(pattern_name='edit_model')),

    # Admin Pages  -------------------------------------------------------------------------
    url(r'^admin/admin_page/$', views.admin_page, name='admin_page'),
    # inspect
    url(r'^admin/inspect_object/$', views.inspect_object, {'object_id': ''}, name='inspect_object_empty'),
    url(r'^admin/inspect_object/(?P<object_id>[-\d]+)/$', views.inspect_object, name='inspect_object'),
    url(r'^admin/get_object/$', views.get_object, name='get_object'),
    url(r'^admin/inspect_upload/$', views.inspect_upload, {'upload_id': ''}, name='inspect_upload_empty'),
    url(r'^admin/inspect_upload/(?P<upload_id>[-\d]+)/$', views.inspect_upload, name='inspect_upload'),
    url(r'^admin/get_uploaded_dataset/$', views.get_uploaded_dataset, name='get_uploaded_dataset'),
    url(r'^admin/inspect_logged_variables/$', views.inspect_logged_variables, name='inspect_logged_variables'),
    url(r'^admin/get_logged_variables_last_x_minutes/$', views.get_logged_variables_last_x_minutes, name='get_logged_variables_last_x_minutes'),
    url(r'^admin/run_query/$', views.run_query, name='run_query'),
    url(r'^admin/get_query_results/$', views.get_query_results, name='get_query_results'),
    # show
    url(r'^admin/show_attributes/$', views.show_attributes, name='show_attributes'),
    url(r'^admin/show_object_types/$', views.show_object_types, name='show_object_types'),
    url(r'^admin/show_newsletter_subscribers/$', views.show_newsletter_subscribers, name='show_newsletter_subscribers'),
    url(r'^admin/show_users/$', views.show_users, name='show_users'),
    # model fixes
    url(r'^admin/salvage_cancelled_simulation/$', views.salvage_cancelled_simulation_page, name='salvage_cancelled_simulation_page'),
    url(r'^admin/salvage_cancelled_simulation/(?P<simulation_id>[-\d]+)/(?P<run_number>[-\d]+)/$', views.salvage_cancelled_simulation, name='salvage_cancelled_simulation'),
    url(r'^admin/show_validation_data/$', views.show_validation_data, name='show_validation_data'),
    url(r'^admin/get_validation_data/$', views.get_validation_data, name='get_validation_data'),
    # data cleaning
    url(r'^admin/possibly_duplicate_objects_without_keys/$', views.possibly_duplicate_objects_without_keys, name='possibly_duplicate_objects_without_keys'),
    url(r'^admin/find_possibly_duplicate_objects_without_keys/$', views.find_possibly_duplicate_objects_without_keys, name='find_possibly_duplicate_objects_without_keys'),
    url(r'^admin/get_possibly_duplicate_objects_without_keys/$', views.get_possibly_duplicate_objects_without_keys, name='get_possibly_duplicate_objects_without_keys'),
    url(r'^admin/possibly_duplicate_objects_with_keys/$', views.possibly_duplicate_objects_with_keys, name='possibly_duplicate_objects_with_keys'),
    url(r'^admin/get_possibly_duplicate_objects_with_keys/$', views.get_possibly_duplicate_objects_with_keys, name='get_possibly_duplicate_objects_with_keys'),
    url(r'^admin/delete_objects_page/$', views.delete_objects_page, name='delete_objects_page'),
    url(r'^admin/delete_objects/$', views.delete_objects, name='delete_objects'),
    url(r'^admin/delete_upload_page/$', views.delete_upload_page, name='delete_upload_page'),
    url(r'^admin/delete_upload/$', views.delete_upload, name='delete_upload'),
    url(r'^admin/delete_simulation_page/$', views.delete_simulation_page, name='delete_simulation_page'),
    url(r'^admin/delete_simulation/$', views.delete_simulation, name='delete_simulation'),
    # various scripts
    url(r'^admin/various_scripts/$', views.various_scripts, name='various_scripts'),
    url(r'^admin/remove_null_datapoints/$', views.remove_null_datapoints, name='remove_null_datapoints'),
    url(r'^admin/remove_duplicate_datapoints/$', views.remove_duplicate_datapoints, name='remove_duplicate_datapoints'),
    url(r'^admin/backup_database/$', views.backup_database, name='backup_database'),
    url(r'^admin/close_db_connections/$', views.close_db_connections, name='close_db_connections'),
    url(r'^admin/clear_database/$', views.clear_database, name='clear_database'),
    url(r'^admin/populate_database/$', views.populate_database, name='populate_database'),

    url(r'^upload_file/$', views.upload_file, name='upload_file'),
    # test pages
    url(r'^test_page1/$', views.test_page1, name='test_page1'),
    url(r'^test_page2/$', views.test_page2, name='test_page2'),
    url(r'^test_page3/$', views.test_page3, name='test_page3'),
    path('admin/', admin.site.urls, name='inspect_database_records'),


]

