####################################################################
# This file is part of the Tree of Knowledge project.
#
# Copyright (c) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, November 2024
#####################################################################


from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.http import Http404
from django.shortcuts import render, redirect
from collection.models import Newsletter_subscriber, Simulation_model, Uploaded_dataset, Object_hierachy_tree_history, Attribute, Object_types, Data_point, Object, Calculation_rule, Learned_rule, Rule, Execution_order, Likelihood_function, Rule_parameter, Logged_variable, Monte_carlo_result, Learn_parameters_result
from django.contrib.auth.models import User
from django.db.models import Count
from collection.forms import UserForm, ProfileForm, Subscriber_preferencesForm, Subscriber_registrationForm, UploadFileForm, Uploaded_datasetForm2, Uploaded_datasetForm3, Uploaded_datasetForm4, Uploaded_datasetForm5, Uploaded_datasetForm6, Uploaded_datasetForm7
from django.template.defaultfilters import slugify
from collection.functions import upload_data, get_from_db, admin_fuctions, tdda_functions, query_datapoints, simulation, generally_useful_functions
from django.http import HttpResponse
import json
import traceback
import csv
import pandas as pd
import xlwt
from django.utils.encoding import smart_str
import time
import os
from django.views.decorators.csrf import csrf_protect, csrf_exempt, requires_csrf_token
import numpy as np
import math
from scipy.stats import beta
import scipy
from django.conf import settings
from django.core.mail import EmailMultiAlternatives
import pdb
from boto import sns
import psycopg2
from datetime import datetime
import scipy.stats as stats
import re



 # ===============================================================================
 #  _______ _           __          __  _         _ _       
 # |__   __| |          \ \        / / | |       (_) |      
 #    | |  | |__   ___   \ \  /\  / /__| |__  ___ _| |_ ___ 
 #    | |  | '_ \ / _ \   \ \/  \/ / _ \ '_ \/ __| | __/ _ \
 #    | |  | | | |  __/    \  /\  /  __/ |_) \__ \ | ||  __/
 #    |_|  |_| |_|\___|     \/  \/ \___|_.__/|___/_|\__\___|
 # 
 # ===============================================================================                                                         

def landing_page(request):
    return render(request, 'landing_page.html')

def about(request):
    return render(request, 'about.html')

def tutorial_overview(request):
    return render(request, 'tutorial_overview.html')

def tutorial1(request):
    return render(request, 'tutorial1.html')

def tutorial2(request):
    return render(request, 'tutorial2.html')

def tutorial3(request):
    return render(request, 'tutorial3.html')

def subscribe(request):
    if request.method == 'POST':
        form_class = Subscriber_registrationForm
        form = form_class(request.POST)

        if form.is_valid():
            form.save()
            email = form.cleaned_data['email']
            first_name = form.cleaned_data['first_name']
            subscriber = Newsletter_subscriber.objects.get(email=email)

            message = '''Hi ''' + first_name + ''',

            Thank you for subscribing to the Tree of Knowledge newsletter.
            '''
            email_message = EmailMultiAlternatives('Tree of Knowledge Newsletter', message, 'noreply@treeofknowledge.ai', [email])
            email_message.send()
            return redirect('subscriber_page', userid=subscriber.userid)
        else:
            return render(request, 'subscribe.html', {'error_occured': True,})
    else:
        return render(request, 'subscribe.html', {'error_occured': False,})


def contact(request):
    return render(request, 'contact.html')




def subscriber_page(request, userid):
    subscriber = Newsletter_subscriber.objects.get(userid=userid)
    is_post_request = False
    if request.method == 'POST':
        is_post_request = True
        form_class = Subscriber_preferencesForm
        form = form_class(data=request.POST, instance=subscriber)
        if form.is_valid():
            form.save()
    return render(request, 'subscriber_page.html', {'subscriber':subscriber, 'is_post_request':is_post_request, })




# ===== THE TOOL ===================================================================
@login_required
def main_menu(request):
    simulation_models = Simulation_model.objects.all().order_by('id') 
    return render(request, 'tool/main_menu.html', {'simulation_models': simulation_models})


@login_required
def open_your_simulation(request):
    simulation_models = Simulation_model.objects.filter(user=request.user).order_by('-id') 
    return render(request, 'tool/open_your_simulation.html', {'simulation_models': simulation_models})

@login_required
def browse_simulations(request):
    simulation_models = Simulation_model.objects.all().order_by('-id') 
    return render(request, 'tool/browse_simulations.html', {'simulation_models': simulation_models})


@login_required
def profile_and_settings(request):
    errors = []
    
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, instance=request.user.profile)
        if not user_form.is_valid():
            errors.append('Error: something is wrong with either the first name, last name or email.')
        else:
            if not profile_form.is_valid():
                errors.append('Error: something is wrong with the message-box setting.')
            else:
                user_form.save()
                profile_form.save()
                return redirect('main_menu')
   
    return render(request, 'tool/profile_and_settings.html', {'errors': errors})



 # ===============================================================
 #   _    _       _                 _       _       _        
 #  | |  | |     | |               | |     | |     | |       
 #  | |  | |_ __ | | ___   __ _  __| |   __| | __ _| |_ __ _ 
 #  | |  | | '_ \| |/ _ \ / _` |/ _` |  / _` |/ _` | __/ _` |
 #  | |__| | |_) | | (_) | (_| | (_| | | (_| | (_| | || (_| |
 #   \____/| .__/|_|\___/ \__,_|\__,_|  \__,_|\__,_|\__\__,_|
 #         | |                                               
 #         |_|         
 # ===============================================================


@login_required
def upload_data1_new(request):
    errors = []
    print('upload_data1_new')
    if request.method == 'POST':
        form1 = UploadFileForm(request.POST, request.FILES)
        if not form1.is_valid():
            errors.append("Error: Form not valid.")
        else:
            data_file = request.FILES['file']
            if data_file.name[-4:] !=".csv":
                errors.append("Error: Uploaded file is not a csv-file.")
            else:
                print('save_new_upload_details')
                (upload_id, upload_error, new_errors) = upload_data.save_new_upload_details(request)
                if upload_error:
                    errors.extend(new_errors)
                    return render(request, 'tool/upload_data1.html', {'upload_error':upload_error, 'errors': errors})
                else:
                    return redirect('upload_data1', upload_id=upload_id)

        return render(request, 'tool/upload_data1.html', {'errors': errors})
        # return redirect('upload_data1', upload_id=upload_id, errors=errors)
    else:
        return render(request, 'tool/upload_data1.html', {'errors': errors})



@login_required
def upload_data1(request, upload_id, errors=[]):
    print('upload_data1')
    # if the upload_id was wrong, send the user back to the first page
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id, user=request.user)
    if uploaded_dataset is None:
        errors.append('Error: %s is not a valid upload_id' % str(upload_id))
        return render(request, 'tool/upload_data1.html', {'errors': errors})

    if request.method == 'POST':
        form1 = UploadFileForm(request.POST, request.FILES)
        if not form1.is_valid():
            errors.append("Error: Form not valid.")
        else:
            data_file = request.FILES['file']
            if data_file.name[-4:] !=".csv":
                errors.append("Error: Uploaded file is not a csv-file.")
            else:
                print('save_existing_upload_details')
                (upload_error, new_errors) = upload_data.save_existing_upload_details(upload_id, request)
                if upload_error:
                    errors.extend(new_errors)
                    return render(request, 'tool/upload_data1.html', {'upload_error':upload_error, 'errors': errors, 'uploaded_dataset':uploaded_dataset})
                else:
                    return redirect('upload_data1', upload_id=upload_id)


    return render(request, 'tool/upload_data1.html', {'uploaded_dataset': uploaded_dataset, 'errors': errors})




@login_required
def upload_data2(request, upload_id):
    errors = []
    error_dict = '{}'
    
    # if the upload_id was wrong, send the user back to the first page
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id, user=request.user)
    if uploaded_dataset is None:
        errors.append('Error: %s is not a valid upload_id' % str(upload_id))
        return render(request, 'tool/upload_data1.html', {'upload_id': upload_id, 'errors': errors})

    if request.method == 'POST':
        form2 = Uploaded_datasetForm2(data=request.POST, instance=uploaded_dataset)
        if not form2.is_valid():
            errors.append('Error: the form is not valid.')
            error_dict = json.dumps(dict(form2.errors))
        else:
            form2.save()
            return redirect('upload_data3', upload_id=upload_id)

    known_data_sources = get_from_db.get_known_data_sources()
    return render(request, 'tool/upload_data2.html', {'upload_id': upload_id, 'uploaded_dataset': uploaded_dataset, 'known_data_sources': known_data_sources, 'errors': errors, 'error_dict': error_dict})


@login_required
def upload_data3(request, upload_id):
    errors = []
    # if the upload_id was wrong, send the user back to the first page
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id, user=request.user)
    if uploaded_dataset is None:
        errors.append('Error: %s is not a valid upload_id' % str(upload_id))
        return render(request, 'tool/upload_data1.html', {'upload_id': upload_id, 'errors': errors})

    
    if request.method == 'POST':
        print('******************************************************')
        print(json.dumps(request.POST))
        print('******************************************************')
        form3 = Uploaded_datasetForm3(data=request.POST, instance=uploaded_dataset)
        if not form3.is_valid():
            errors.append('Error: the form is not valid.')
        else:
            form3.save()
            return redirect('upload_data4', upload_id=upload_id)

    object_hierachy_tree = get_from_db.get_object_hierachy_tree()
    return render(request, 'tool/upload_data3.html', {'upload_id': upload_id, 'uploaded_dataset': uploaded_dataset, 'object_hierachy_tree':object_hierachy_tree, 'errors': errors})


@login_required
def upload_data4(request, upload_id):
    errors = []
    # if the upload_id was wrong, send the user back to the first page
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id, user=request.user)
    if uploaded_dataset is None:
        errors.append('Error: %s is not a valid upload_id' % str(upload_id))
        return render(request, 'tool/upload_data1.html', {'upload_id': upload_id, 'errors': errors})

    
    if request.method == 'POST':
        meta_data_facts_old = json.loads(request.POST['meta_data_facts'])
        meta_data_facts_new = get_from_db.convert_fact_values_to_the_right_format(meta_data_facts_old)
        request.POST._mutable = True
        request.POST['meta_data_facts'] = json.dumps(meta_data_facts_new)

        form4 = Uploaded_datasetForm4(data=request.POST, instance=uploaded_dataset)
        if not form4.is_valid():
            errors.append('Error: the form is not valid.')
        else:
            form4.save()
            return redirect('upload_data5', upload_id=upload_id)

    data_generation_year = "2015"
    if uploaded_dataset.data_generation_date is not None:
        data_generation_year = uploaded_dataset.data_generation_date.year
    return render(request, 'tool/upload_data4.html', {'upload_id': upload_id, 'uploaded_dataset': uploaded_dataset, 'data_generation_year':data_generation_year, 'errors': errors})




@login_required
def upload_data5(request, upload_id):
    errors = []
    # if the upload_id was wrong, send the user back to the first page
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id, user=request.user)
    if uploaded_dataset is None:
        errors.append('Error: %s is not a valid upload_id' % str(upload_id))
        return render(request, 'tool/upload_data1.html', {'upload_id': upload_id, 'errors': errors})
    
    if request.method == 'POST':
        form5 = Uploaded_datasetForm5(data=request.POST, instance=uploaded_dataset)
        if not form5.is_valid():
            errors.append('Error: the form is not valid.')
        else:
            form5.save()
            if request.POST.get('datetime_column') is None:
                return redirect('upload_data6A', upload_id=upload_id) #non-timeseries
            else:
                return redirect('upload_data6B', upload_id=upload_id) #timeseries
 
    return render(request, 'tool/upload_data5.html', {'upload_id': upload_id, 'uploaded_dataset': uploaded_dataset, 'errors': errors})



@login_required
def upload_data6A(request, upload_id):
    errors = []
    # if the upload_id was wrong, send the user back to the first page
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id, user=request.user)
    if uploaded_dataset is None:
        errors.append('Error: %s is not a valid upload_id' % str(upload_id))
        return render(request, 'tool/upload_data1.html', {'upload_id': upload_id, 'errors': errors})

    if request.method == 'POST':
        form6 = Uploaded_datasetForm6(data=request.POST, instance=uploaded_dataset)
        if not form6.is_valid():
            errors.append('Error: the form is not valid.')
        else:
            form6.save()
            if uploaded_dataset.data_generation_date is None:
                return redirect('upload_data7', upload_id=upload_id)
            else:
                (number_of_datapoints_saved, new_model_id) = upload_data.perform_uploading(uploaded_dataset, request)
                return redirect('upload_data_success', number_of_datapoints_saved=number_of_datapoints_saved, new_model_id=new_model_id)
   
    table_attributes = upload_data.make_table_attributes_dict(uploaded_dataset)
    return render(request, 'tool/upload_data6.html', {'upload_id': upload_id, 'data_table_json': uploaded_dataset.data_table_json, 'table_attributes': table_attributes, 'errors': errors})



@login_required
def upload_data6B(request, upload_id):
    errors = []
    # if the upload_id was wrong, send the user back to the first page
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id, user=request.user)
    if uploaded_dataset is None:
        errors.append('Error: %s is not a valid upload_id' % str(upload_id))
        return render(request, 'tool/upload_data1.html', {'upload_id': upload_id, 'errors': errors})

    if request.method == 'POST':
        form6 = Uploaded_datasetForm6(data=request.POST, instance=uploaded_dataset)
        print(request.POST.get('list_of_matches', None))
        print(request.POST.get('upload_only_matched_entities', None))
        
        if not form6.is_valid():
            errors.append('Error: the form is not valid.')
        else:
            form6.save()
            (number_of_datapoints_saved, new_model_id) = upload_data.perform_uploading_for_timeseries(uploaded_dataset, request)
            if 'DB_CONNECTION_URL' in os.environ:
                return redirect('https://www.treeofknowledge.ai/tool/upload_data_success', number_of_datapoints_saved=number_of_datapoints_saved, new_model_id=new_model_id)
            else:
                return redirect('upload_data_success', number_of_datapoints_saved=number_of_datapoints_saved, new_model_id=new_model_id)
   
    
    table_attributes = upload_data.make_table_attributes_dict(uploaded_dataset)
    if uploaded_dataset.object_identifiers is None:
        data_table_json = uploaded_dataset.data_table_json
    else: 
        data_table_json_dict = upload_data.make_data_table_json_with_distinct_entities(uploaded_dataset)
        data_table_json = json.dumps(data_table_json_dict)
    return render(request, 'tool/upload_data6.html', {'upload_id': upload_id, 'data_table_json': data_table_json, 'table_attributes': table_attributes, 'errors': errors})




@login_required
def upload_data7(request, upload_id):
    errors = []
    # if the upload_id was wrong, send the user back to the first page
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id, user=request.user)
    if uploaded_dataset is None:
        errors.append('Error: %s is not a valid upload_id' % str(upload_id))
        return render(request, 'tool/upload_data1.html', {'upload_id': upload_id, 'errors': errors})

    
    if request.method == 'POST':
        form7 = Uploaded_datasetForm7(data=request.POST, instance=uploaded_dataset)
        if not form7.is_valid():
            errors.append('Error: the form is not valid.')
        else:
            form7.save()
            (number_of_datapoints_saved, new_model_id) = upload_data.perform_uploading(uploaded_dataset, request)
            if 'DB_CONNECTION_URL' in os.environ:
                return redirect('https://www.treeofknowledge.ai/tool/upload_data_success', number_of_datapoints_saved=number_of_datapoints_saved, new_model_id=new_model_id)
            else:
                return redirect('upload_data_success', number_of_datapoints_saved=number_of_datapoints_saved, new_model_id=new_model_id)
   
    return render(request, 'tool/upload_data7.html', {'upload_id': upload_id, 'uploaded_dataset': uploaded_dataset, 'errors': errors})


@login_required
def upload_data_success(request, number_of_datapoints_saved, new_model_id):
    all_simulation_models = Simulation_model.objects.all().order_by('id') 
    return render(request, 'tool/upload_data_success.html', {'number_of_datapoints_saved':number_of_datapoints_saved, 'new_model_id':new_model_id, 'all_simulation_models': all_simulation_models})



@login_required
def get_upload_progress(request):
    upload_id = request.GET.get('upload_id', '')

    with open('collection/static/webservice files/runtime_data/upload_progress_' + upload_id + '.txt') as file:       
        progress = file.readline().strip()

    return HttpResponse(progress)



 # =================================================================================
 #  _    _      _                   ______                _   _                 
 # | |  | |    | |                 |  ____|              | | (_)                
 # | |__| | ___| |_ __   ___ _ __  | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
 # |  __  |/ _ \ | '_ \ / _ \ '__| |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
 # | |  | |  __/ | |_) |  __/ |    | |  | |_| | | | | (__| |_| | (_) | | | \__ \
 # |_|  |_|\___|_| .__/ \___|_|    |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
 #               | |                                                            
 #               |_|                                                          
 # =================================================================================               

# ==================
# simple GET
# ==================


@login_required
def get_possible_attributes(request):
    object_type_id = request.GET.get('object_type_id', '')
    list_of_parent_objects = get_from_db.get_list_of_parent_objects(object_type_id)
    list_of_parent_object_ids = [el['id'] for el in list_of_parent_objects]



    response = []
    attributes = Attribute.objects.all().filter(first_applicable_object_type__in=list_of_parent_object_ids)
    
    for attribute in attributes:
        response.append({'attribute_id': attribute.id, 'attribute_name': attribute.name, 'attribute_data_type': attribute.data_type, 'attribute_first_relation_object_type': attribute.first_relation_object_type})
    return HttpResponse(json.dumps(response))

# used in create_attribute_modal.html
@login_required
def get_list_of_parent_objects(request):
    object_type_id = request.GET.get('object_type_id', '')
    list_of_parent_objects = get_from_db.get_list_of_parent_objects(object_type_id)
    print('___________________________________________________________')
    print(object_type_id)
    print(list_of_parent_objects)
    print('___________________________________________________________')
    return HttpResponse(json.dumps(list_of_parent_objects))


# used in edit_attribute_modal.html
@login_required
def get_list_of_objects(request):
    list_of_objects = []
    object_records = Object_types.objects.all()
    for object_record in object_records:
        list_of_objects.append({'id':object_record.id, 'name':object_record.name})
    return HttpResponse(json.dumps(list_of_objects))    


# used in edit_attribute_modal.html
@login_required
def get_attribute_details(request):
    attribute_id = request.GET.get('attribute_id', '')
    attribute_id = int(attribute_id)
    attribute_record = Attribute.objects.get(id=attribute_id)
    attribute_details = {'id':attribute_record.id, 
                'name':attribute_record.name,
                'data_type':attribute_record.data_type,
                'expected_valid_period':attribute_record.expected_valid_period,
                'description':attribute_record.description,
                'format_specification':json.loads(attribute_record.format_specification),
                'first_applicable_object_type':attribute_record.first_applicable_object_type,
                'first_relation_object_type':attribute_record.first_relation_object_type}
    print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    print(str(attribute_id))
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    return HttpResponse(json.dumps(attribute_details))   


# used in edit_model.html
@login_required
def get_attribute_rules_old(request):
    attribute_id = request.GET.get('attribute_id', '')
    attribute_id = int(attribute_id)
    rule_records = Calculation_rule.objects.filter(attribute_id=attribute_id).order_by('-number_of_times_used')
    rules_list = list(rule_records.values())
    return HttpResponse(json.dumps(rules_list)) 



@login_required
def get_object_hierachy_tree(request):
    object_hierachy_tree = get_from_db.get_object_hierachy_tree()
    return HttpResponse(object_hierachy_tree)


# @login_required
# def get_available_variables(request):
#     object_type_id = request.GET.get('object_type_id', '')
#     available_variables = []
#     list_of_parent_objects = get_from_db.get_list_of_parent_objects(object_type_id)
#     for parent_object in list_of_parent_objects:
#         available_variables.extend(list(Attribute.objects.filter(first_applicable_object_type=parent_object['id']).values('name', 'id', 'data_type')))
#     return HttpResponse(json.dumps(available_variables))
    


# used in edit_simulation__simulate.html
@login_required
def get_object_rules(request):
    print('----------- get_object_rules -------------')
    execution_order_id = int(request.GET.get('execution_order_id', ''))
    object_number = request.GET.get('object_number', '')
    object_type_id = request.GET.get('object_type_id', '')

    response = {'object_number': int(object_number), 'object_rules':{}}
    print(object_type_id)
    list_of_parent_objects = get_from_db.get_list_of_parent_objects(object_type_id)
    parent_object_type_ids = [obj['id'] for obj in list_of_parent_objects]
    attributes = Attribute.objects.filter(first_applicable_object_type__in=parent_object_type_ids).values()
    
    print('attribute_ids: ' + str([attribute['id'] for attribute in list(attributes)]))
    for attribute in attributes:
        response['object_rules'][attribute['id']] = {}
        rules_list = list(Rule.objects.filter(changed_var_attribute_id=attribute['id']).values())
        for rule in rules_list:
            rule['used_attribute_ids'] = json.loads(rule['used_attribute_ids'])
            rule['used_parameter_ids'] = json.loads(rule['used_parameter_ids'])

            # calculate 'probability' and 'standard_dev'
            histogram, mean, standard_dev, nb_of_simulations, nb_of_sim_in_which_rule_was_used, nb_of_tested_parameters, nb_of_tested_parameters_in_posterior, histogram_smooth = get_from_db.get_rules_pdf(execution_order_id, rule['id'], True)
            rule['probability'] = None if mean is None or np.isnan(mean) else mean 
            rule['standard_dev'] = None if standard_dev is None or np.isnan(standard_dev) else standard_dev 


            # specify a default for 'learn_posterior'
            rule['learn_posterior'] = False
            if not rule['has_probability_1'] and rule['probability'] is None:
                rule['learn_posterior'] = True

            response['object_rules'][attribute['id']][rule['id']] = rule
    return HttpResponse(json.dumps(response)) 



@login_required
def get_all_pdfs(request):
    print('get_all_pdfs 1')
    from django.db import connection
    execution_order_id = int(request.GET.get('execution_order_id', ''))
    rule_or_parameter_id = int(request.GET.get('rule_or_parameter_id', ''))
    is_rule = (request.GET.get('is_rule', '').lower() == 'true')
    response = {}


    response = {'execution_order_name': Execution_order.objects.get(id=execution_order_id).name,'rule_or_parameter_id':rule_or_parameter_id, 'is_rule':is_rule, 'individual_pdfs':[]}
    print('get_all_pdfs 2')
    if is_rule:
        query = ''' SELECT  DISTINCT simulation_id, object_number, 
                            FIRST_VALUE(list_of_probabilities) over (partition by simulation_id, object_number ORDER BY id DESC) as list_of_probabilities, 
                            FIRST_VALUE(nb_of_simulations) over (partition by simulation_id, object_number ORDER BY id DESC) as nb_of_simulations, 
                            FIRST_VALUE(nb_of_sim_in_which_rule_was_used) over (partition by simulation_id, object_number ORDER BY id DESC) as nb_of_sim_in_which_rule_was_used, 
                            FIRST_VALUE(nb_of_tested_parameters) over (partition by simulation_id, object_number ORDER BY id DESC) as nb_of_tested_parameters, 
                            FIRST_VALUE(nb_of_tested_parameters_in_posterior) over (partition by simulation_id, object_number ORDER BY id DESC) as nb_of_tested_parameters_in_posterior 
                    FROM collection_likelihood_function 
                    WHERE rule_id=%s  
                      AND execution_order_id=%s
                      AND nb_of_tested_parameters_in_posterior > 0 
                    ORDER BY simulation_id; ''' % (rule_or_parameter_id, execution_order_id)

    else: 
        query = ''' SELECT  distinct simulation_id, object_number, 
                            FIRST_VALUE(list_of_probabilities) over (partition by simulation_id, object_number order by id DESC) as list_of_probabilities, 
                            FIRST_VALUE(nb_of_simulations) over (partition by simulation_id, object_number order by id DESC) as nb_of_simulations, 
                            FIRST_VALUE(nb_of_sim_in_which_rule_was_used) over (partition by simulation_id, object_number order by id DESC) as nb_of_sim_in_which_rule_was_used, 
                            FIRST_VALUE(nb_of_tested_parameters) over (partition by simulation_id, object_number order by id DESC) as nb_of_tested_parameters, 
                            FIRST_VALUE(nb_of_tested_parameters_in_posterior) over (partition by simulation_id, object_number order by id DESC) as nb_of_tested_parameters_in_posterior 
                    FROM collection_likelihood_function 
                    WHERE parameter_id=%s  
                      AND execution_order_id=%s
                      AND nb_of_tested_parameters_in_posterior > 0 
                    ORDER BY simulation_id; ''' % (rule_or_parameter_id, execution_order_id)

    individual_pdfs_df = pd.read_sql_query(query, connection)



    print('get_all_pdfs 3')
    for index, row in individual_pdfs_df.iterrows():
        print('get_all_pdfs 3.1')
        list_of_probabilities = np.asarray(json.loads(row['list_of_probabilities']))
        list_of_probabilities = [np.float64(el) for el in list_of_probabilities]
        list_of_probabilities_smooth = np.zeros(30)
        nb_of_tested_parameters_in_posterior = row['nb_of_tested_parameters_in_posterior']

        # kernel smoothing for list_of_probabilities_smooth
        print('get_all_pdfs 3.2')
        x = np.linspace(-1, 1, 59)
        sigma = 0.03 + 1/(nb_of_tested_parameters_in_posterior)
        weights = stats.norm.pdf(x, 0, sigma)
        for position in range(30):
            list_of_probabilities_smooth[position] = np.sum(list_of_probabilities * weights[29-position:59-position])
        list_of_probabilities_smooth = list_of_probabilities_smooth * 30/ np.sum(list_of_probabilities_smooth) 
        


        print('get_all_pdfs 3.3')
        try:
            simulation_name = Simulation_model.objects.get(id=row['simulation_id']).simulation_name
        except:
            simulation_name = "This simulation has already been deleted."
        response['individual_pdfs'].append({'simulation_id': row['simulation_id'], 
                                            'simulation_name':simulation_name, 
                                            'nb_of_tested_parameters': row['nb_of_tested_parameters'],
                                            'nb_of_tested_parameters_in_posterior': row['nb_of_tested_parameters_in_posterior'],
                                            'pdf': [[bucket_value, count] for bucket_value, count in zip(list(np.linspace(0,1,31)), list_of_probabilities)],
                                            'smooth_pdf': [[bucket_value, count] for bucket_value, count in zip(list(np.linspace(0,1,31)), list_of_probabilities_smooth)],
                                            })
    
    print('get_all_pdfs 4')  
    histogram, mean, standard_dev, nb_of_simulations, nb_of_sim_in_which_rule_was_used, nb_of_tested_parameters, nb_of_tested_parameters_in_posterior, histogram_smooth = get_from_db.get_rules_pdf(execution_order_id, rule_or_parameter_id, is_rule)
    response['combined_pdf'] = {'nb_of_tested_parameters': nb_of_tested_parameters,
                                'nb_of_tested_parameters_in_posterior': nb_of_tested_parameters_in_posterior,
                                'pdf': [[bucket_value, min(count,10000)] for bucket_value, count in zip(histogram[1], histogram[0])],
                                'smooth_pdf': [[bucket_value, min(count,10000)] for bucket_value, count in zip(histogram_smooth[1], histogram_smooth[0])]
                                }
    print('get_all_pdfs 5')
    return HttpResponse(json.dumps(response))




# used in edit_object_behaviour_modal.html (which in turn is used in edit_simulation.html and analyse_simulation.html)
@login_required
def get_rules_pdf(request):
    print('====================   get_rules_pdf   ==================================')
    execution_order_id = int(request.GET.get('execution_order_id', ''))
    rule_or_parameter_id = int(request.GET.get('rule_or_parameter_id', ''))
    is_rule = (request.GET.get('is_rule', '').lower() == 'true')

    histogram, mean, standard_dev, nb_of_simulations, nb_of_sim_in_which_rule_was_used, nb_of_tested_parameters, nb_of_tested_parameters_in_posterior, histogram_smooth = get_from_db.get_rules_pdf(execution_order_id, rule_or_parameter_id, is_rule)
    response = {'nb_of_simulations': nb_of_simulations,
                'nb_of_sim_in_which_rule_was_used': nb_of_sim_in_which_rule_was_used,
                'nb_of_tested_parameters': nb_of_tested_parameters,
                'nb_of_tested_parameters_in_posterior': nb_of_tested_parameters_in_posterior}

    if histogram is None:
        return HttpResponse('null')
    

    response['pdf'] = [[bucket_value, min(count,10000)] for bucket_value, count in zip(histogram_smooth[1], histogram_smooth[0])]
    return HttpResponse(json.dumps(response))


# used in edit_object_behaviour_modal.html (which in turn is used in edit_simulation.html and analyse_simulation.html)
@login_required
def get_single_pdf(request):
    response = {}
    simulation_id = request.GET.get('simulation_id', '')
    execution_order_id = request.GET.get('execution_order_id', '')
    object_number = request.GET.get('object_number', '')
    rule_or_parameter_id = request.GET.get('rule_or_parameter_id', '')
    is_rule = (request.GET.get('is_rule', '').lower() == 'true')
    histogram, mean, standard_dev, nb_of_sim_in_which_rule_was_used, message = get_from_db.get_single_pdf(simulation_id, execution_order_id, object_number, rule_or_parameter_id, is_rule)
    print('====================   get_single_pdf   ==================================')
    print(str(rule_or_parameter_id))
    print(str(simulation_id))
    print(str(object_number))
    print(str(rule_or_parameter_id))
    print(str(is_rule))
    print(str(histogram is None))
    print(str(histogram))
    print('=========================================================================')
    response['nb_of_sim_in_which_rule_was_used'] = nb_of_sim_in_which_rule_was_used
    if message != '':
        response['message'] = message

    if histogram is None:
        return HttpResponse('null')


    response['pdf'] = [[bucket_value, count] for bucket_value, count in zip(histogram[1], histogram[0])]
    return HttpResponse(json.dumps(response))
    


# used in edit_object_type_behaviour.html
@login_required
def get_parameter_info(request):
    response = {}
    rule_parameters = Rule_parameter.objects.filter().all()
    for rule_parameter in rule_parameters:
        response[rule_parameter.id] =  {'id': rule_parameter.id,
                                        'rule_id':rule_parameter.rule_id, 
                                        'parameter_name':rule_parameter.parameter_name,
                                        'min_value':rule_parameter.min_value,
                                        'max_value':rule_parameter.max_value}

    return HttpResponse(json.dumps(response))   




# used in analyse_learned_parameters.html
@login_required
def get_simulated_parameter_numbers(request):
    simulation_id = request.GET.get('simulation_id', '')
    execution_order_id = request.GET.get('execution_order_id', '')
    run_number = request.GET.get('run_number', '')
    learned_parameter_numbers = Monte_carlo_result.objects.filter(simulation_id=simulation_id, execution_order_id=execution_order_id, run_number=run_number, is_new_parameter=False).order_by('parameter_number').values_list('parameter_number', flat=True)
    learned_parameter_numbers = list(set(learned_parameter_numbers))

    new_parameter_numbers = Monte_carlo_result.objects.filter(simulation_id=simulation_id, execution_order_id=execution_order_id, run_number=run_number, is_new_parameter=True).order_by('parameter_number').values_list('parameter_number', flat=True)
    new_parameter_numbers = list(set(new_parameter_numbers))
    return HttpResponse(json.dumps({'learned_parameter_numbers':learned_parameter_numbers, 'new_parameter_numbers':new_parameter_numbers}))  



# used in analyse_simulation.html
@login_required
def get_missing_objects_dict_attributes(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            response = {}
            for object_number in request_body.keys():
                response[object_number] = {}
                simulation_data_columns = request_body[object_number]['simulation_data_columns']
                simulation_data_attribute_ids = [int(re.findall(r'\d+', col)[1]) for col in simulation_data_columns if len(re.findall(r'\d+', col))>1]
                objects_dict_attribute_ids = [int(attribute_id) for attribute_id in request_body[object_number]['objects_dict_attribute_ids']]

                missing_attribute_ids = list(set(simulation_data_attribute_ids) - set(objects_dict_attribute_ids))
                for missing_attribute_id in missing_attribute_ids:

                    attribute_record = Attribute.objects.get(id=missing_attribute_id)
                    response[object_number][missing_attribute_id] =  {  'attribute_value': None, 
                                                                        'attribute_name':attribute_record.name, 
                                                                        'attribute_data_type':attribute_record.data_type, 
                                                                        'attribute_rule': None}

            return HttpResponse(json.dumps(response))
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")



# used in analyse_learned_parameters.html
@login_required
def get_all_priors_df_and_learned_rules(request):
    simulation_id = int(request.GET.get('simulation_id', ''))
    execution_order_id = int(request.GET.get('execution_order_id', ''))
    run_number = int(request.GET.get('run_number', ''))

    learn_parameters_result = Learn_parameters_result.objects.filter(simulation_id=simulation_id, execution_order_id=execution_order_id, run_number=run_number).order_by('-id').first()           
    if learn_parameters_result is None:
        return HttpResponse("doesn't exist")  
    else:    
        response = {'all_priors_df': json.loads(learn_parameters_result.all_priors_df), 'learned_rules': json.loads(learn_parameters_result.learned_rules)}
        return HttpResponse(json.dumps(response).replace('": NaN', '": null'))  



# ==================
# check
# ==================

# used in edit_simulation.html
@login_required
def check_if_simulation_results_exist(request):
    simulation_id = int(request.GET.get('simulation_id', ''))
    execution_order_id = int(request.GET.get('execution_order_id', ''))

    simulation_results_exist = Learn_parameters_result.objects.filter(simulation_id=simulation_id, execution_order_id=execution_order_id).count() > 0           
    return HttpResponse(json.dumps(simulation_results_exist))  



# ==================
# complex GET
# ==================

# used in query_data.html
@login_required
def get_data_points(request):
    request_body = json.loads(request.body)
    object_type_id = request_body['object_type_id']
    filter_facts = request_body['filter_facts']
    specified_start_time = int(request_body['specified_start_time'])
    specified_end_time = int(request_body['specified_end_time'])

    response = query_datapoints.get_data_points(object_type_id, filter_facts, specified_start_time, specified_end_time)
    return HttpResponse(json.dumps(response))


# used in edit_model.html
@login_required
def get_data_from_random_object(request):
    request_body = json.loads(request.body)
    object_number = request_body['object_number']
    object_type_id = request_body['object_type_id']
    filter_facts = request_body['filter_facts']
    specified_start_time = request_body['specified_start_time']
    specified_end_time = request_body['specified_end_time']

    (object_id, attribute_values) = query_datapoints.get_data_from_random_object(object_type_id, filter_facts, specified_start_time, specified_end_time)
    response = {'object_number':object_number, 'object_id':object_id, 'attribute_values':attribute_values}
    return HttpResponse(json.dumps(response))


# used in edit_model.html
@login_required
def get_data_from_random_related_object(request):
    request_body = json.loads(request.body)
    simulation_id = request_body['simulation_id']
    objects_dict = request_body['objects_dict']
    environment_start_time = request_body['environment_start_time']
    environment_end_time = request_body['environment_end_time']
    max_number_of_instances = request_body['max_number_of_instances']


    objects_data = query_datapoints.get_data_from_random_related_object(simulation_id, objects_dict, environment_start_time, environment_end_time, max_number_of_instances)
    data_querying_info = Simulation_model.objects.get(id=simulation_id).data_querying_info
    print('preparing response')
    response = {'objects_data': objects_data, 
                'data_querying_info': data_querying_info}    
    print('sending response')           
    return HttpResponse(json.dumps(response))


# used in query_data.html
@login_required
def get_data_from_objects_behind_the_relation(request):
    request_body = json.loads(request.body)
    object_type_id = request_body['object_type_id']
    object_ids = request_body['object_ids']
    object_ids = list(set([el for el in object_ids if el])) #distinct not-null values
    specified_start_time = request_body['specified_start_time']
    specified_end_time = request_body['specified_end_time']
    print('*******************************************************')
    print(request_body.keys())
    print(request_body['object_type_id'])
    print(request_body['object_ids'])
    print(request_body['specified_start_time'])
    print(request_body['specified_end_time'])
    print('*******************************************************')

    response = query_datapoints.get_data_from_objects_behind_the_relation(object_type_id, object_ids, specified_start_time, specified_end_time)   
    return HttpResponse(json.dumps(response))




@login_required
def get_execution_order(request):
    
    print('------------------ get_execution_order --------------------------------')
    execution_order_id = int(request.GET.get('execution_order_id', '1'))
    print('test')
    execution_order = json.loads(Execution_order.objects.get(id=execution_order_id).execution_order)
    if 'attribute_execution_order' not in execution_order.keys():
        execution_order['attribute_execution_order'] = {}
    if 'rule_execution_order' not in execution_order.keys():
        execution_order['rule_execution_order'] = {}
    # CORRECT TO CURRENT objects, attributes and rules

    # PART 1A: extend with missing attributes 
    all_object_type_ids = [el[0] for el in list(Object_types.objects.all().values_list('id'))]
    for object_type_id in all_object_type_ids:
        list_of_parent_objects = get_from_db.get_list_of_parent_objects(object_type_id)
        list_of_parent_object_ids = [el['id'] for el in list_of_parent_objects]
        all_attributes = list(Attribute.objects.all().filter(first_applicable_object_type__in=list_of_parent_object_ids).values('id', 'name'))

        
        if object_type_id not in execution_order['attribute_execution_order'].keys():
            execution_order['attribute_execution_order'][object_type_id] = {'used_attributes':all_attributes, 'not_used_attributes': []}
        else:
            listed_attributes = execution_order['attribute_execution_order'][object_type_id]['used_attributes'] + execution_order['attribute_execution_order'][object_type_id]['not_used_attributes']
            listed_attribute_ids = set([attribute['id'] for attribute in listed_attributes])
            all_attribute_ids = set([attribute['id'] for attribute in all_attributes])
            if len(listed_attribute_ids) < len(all_attribute_ids):
                missing_attribute_ids = list(all_attribute_ids - listed_attribute_ids)
                missing_attributes = [attribute for attribute in all_attributes if attribute['id'] in missing_attribute_ids]
                execution_order['attribute_execution_order'][object_type_id]['used_attributes'] += missing_attributes

    # PART 1B: extend with missing rules
    all_attribute_ids = [el[0] for el in list(Attribute.objects.all().values_list('id'))]
    for attribute_id in all_attribute_ids:
        all_rule_ids = [el[0] for el in list(Rule.objects.all().filter(changed_var_attribute_id=attribute_id).values_list('id'))]

        if (str(attribute_id) not in execution_order['rule_execution_order'].keys()):
            execution_order['rule_execution_order'][str(attribute_id)] = {'used_rule_ids': [], 'not_used_rule_ids': all_rule_ids}
        else:
            listed_rule_ids = set(execution_order['rule_execution_order'][str(attribute_id)]['used_rule_ids'] + execution_order['rule_execution_order'][str(attribute_id)]['not_used_rule_ids'])
            if len(listed_rule_ids) < len(all_rule_ids):
                missing_rule_ids = list(set(all_rule_ids) - listed_rule_ids)
                execution_order['rule_execution_order'][str(attribute_id)]['not_used_rule_ids'] += missing_rule_ids


    # PART 2A: remove no-longer-existing objects
    no_longer_existing_object_type_ids = set(execution_order['attribute_execution_order'].keys()) - set(all_object_type_ids)
    for no_longer_existing_object_type_id in no_longer_existing_object_type_ids:
        del execution_order['attribute_execution_order'][no_longer_existing_object_type_id]

    # PART 2B: remove no-longer-existing attributes 
    for object_type_id in all_object_type_ids:
        # used_attributes
        used_attributes = execution_order['attribute_execution_order'][object_type_id]['used_attributes']
        used_attribute_ids = [attribute['id'] for attribute in used_attributes]
        no_longer_existing_attribute_ids = set(used_attribute_ids) - set(all_attribute_ids)
        if len(no_longer_existing_attribute_ids)>0:
            new_used_attributes = [attribute for attribute in used_attributes if attribute['id'] not in no_longer_existing_attribute_ids]
            execution_order['attribute_execution_order'][object_type_id]['used_attributes'] = new_used_attributes

        # not_used_attributes
        not_used_attributes = execution_order['attribute_execution_order'][object_type_id]['not_used_attributes']
        not_used_attribute_ids = [attribute['id'] for attribute in not_used_attributes]
        no_longer_existing_attribute_ids = set(not_used_attribute_ids) - set(all_attribute_ids)
        if len(no_longer_existing_attribute_ids)>0:
            new_not_used_attributes = [attribute for attribute in not_used_attributes if attribute['id'] not in no_longer_existing_attribute_ids]
            execution_order['attribute_execution_order'][object_type_id]['not_used_attributes'] = new_not_used_attributes

    # PART 2C: remove no-longer-existing attributes 
    all_attribute_ids = [str(el) for el in all_attribute_ids]
    no_longer_existing_attribute_ids = set(execution_order['rule_execution_order'].keys()) - set(all_attribute_ids)
    for no_longer_existing_attribute_id in no_longer_existing_attribute_ids:
        del execution_order['rule_execution_order'][no_longer_existing_attribute_id]


    # PART 2D: remove no-longer-existing rules 
    all_rule_ids = [el[0] for el in list(Rule.objects.all().values_list('id'))]
    for attribute_id in execution_order['rule_execution_order'].keys():
        # used_rule_ids
        used_rule_ids = execution_order['rule_execution_order'][attribute_id]['used_rule_ids']
        no_longer_existing_rule_ids = set(used_rule_ids) - set(all_rule_ids)
        if len(no_longer_existing_rule_ids)>0:
            new_used_rule_ids = [rule_id for rule_id in used_rule_ids if rule_id not in no_longer_existing_rule_ids]
            execution_order['rule_execution_order'][attribute_id]['used_rule_ids'] = new_used_rule_ids

        # not_used_rule_ids
        not_used_rule_ids = execution_order['rule_execution_order'][attribute_id]['not_used_rule_ids']
        no_longer_existing_rule_ids = set(not_used_rule_ids) - set(all_rule_ids)
        if len(no_longer_existing_rule_ids)>0:
            new_not_used_rule_ids = [rule_id for rule_id in not_used_rule_ids if rule_id not in no_longer_existing_rule_ids]
            execution_order['rule_execution_order'][attribute_id]['not_used_rule_ids'] = new_not_used_rule_ids

    # PART 3: update the database
    execution_order_record = Execution_order.objects.get(id=execution_order_id)
    execution_order_record.execution_order = json.dumps(execution_order)
    execution_order_record.save()

    return HttpResponse(json.dumps(execution_order))






# ==================
# FIND
# ==================


# used in: upload_data5
@login_required
def find_suggested_attributes(request):

    request_body = json.loads(request.body)
    attributenumber = request_body['attributenumber']
    object_type_id = request_body['object_type_id']
    column_values = request_body['column_values']

    response = []
    attributes = Attribute.objects.all()
    for attribute in attributes:
        format_specification = json.loads(attribute.format_specification)
        attribute_format = format_specification['fields']['column']
        response.append({'attribute_id': attribute.id, 'attribute_name': attribute.name, 'description': attribute.description, 'format': attribute_format, 'comments': '{}'})
    return HttpResponse(json.dumps(response))


# used in: upload_data5 
# get_suggested_attributes2 get the concluding_format instead of just the attribute's format
@login_required
def find_suggested_attributes2(request):
    print('find_suggested_attributes2')
    request_body = json.loads(request.body)
    attributenumber = request_body['attributenumber']
    object_type_id = request_body['object_type_id']
    upload_id = int(request_body['upload_id'])
    column_values = request_body['column_values']
    print('1')

    list_of_parent_objects = get_from_db.get_list_of_parent_objects(object_type_id)
    list_of_parent_object_ids = [el['id'] for el in list_of_parent_objects]
  
    print('2')
    response = []
    attributes = Attribute.objects.all().filter(first_applicable_object_type__in=list_of_parent_object_ids)

    print('3')
    for attribute in attributes:
        print('4')
        concluding_format = get_from_db.get_attributes_concluding_format(attribute.id, object_type_id, upload_id)
        print('5')
        response.append({'attribute_id': attribute.id, 'attribute_name': attribute.name, 'description': attribute.description, 'format': concluding_format['format_specification'], 'comments': concluding_format['comments'], 'data_type': attribute.data_type, 'object_type_id_of_related_object': attribute.first_relation_object_type})

    print('6')
    return HttpResponse(json.dumps(response))


# used in: upload_data6 
@login_required
def find_matching_entities(request):
    request_body = json.loads(request.body)
    match_attributes = request_body['match_attributes']
    match_values = request_body['match_values']
    matching_objects_entire_list_string = query_datapoints.find_matching_entities(match_attributes, match_values)
    print(matching_objects_entire_list_string)
    print(type(matching_objects_entire_list_string))
    return HttpResponse(matching_objects_entire_list_string)


# used in: additional_facts_functions.html, which in turn is used in upload_data3 and upload_data4
# this function should be extended to also find fuzzy matches and suggest them in the format_violation_text
@login_required
def find_single_entity(request):
    relation_id = int(request.GET.get('relation_id', ''))
    print('++++++++ ' + str(relation_id) + ' +++++++++++++++++')
    attribute_id = request.GET.get('attribute_id', '')
    value = request.GET.get('value', '')

    matching_object_id = query_datapoints.find_single_entity(relation_id, attribute_id, value)
    print(str(matching_object_id))
    response = {'fact_number': int(request.GET.get('fact_number', '')),
                'matching_object_id':matching_object_id}
    return HttpResponse(json.dumps(response))


# ==================
# SAVE
# ==================
@login_required
def save_new_object_hierachy_tree(request):
    if request.method == 'POST':
        new_entry = Object_hierachy_tree_history(object_hierachy_tree=request.body, user=request.user)
        new_entry.save()
        return HttpResponse("success")
    else:
        return HttpResponse("This must be a POST request.")


# used in: upload_data3 (the create-object-type modal)
@login_required
def save_new_object_type(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            object_facts = request_body['li_attr']['attribute_values']
            request_body['li_attr']['attribute_values'] = get_from_db.convert_fact_values_to_the_right_format(object_facts)
            new_entry = Object_types(id=request_body['id'], parent=request_body['parent'], name=request_body['text'], li_attr=json.dumps(request_body['li_attr']), a_attr=None, object_type_icon="si-glyph-square-dashed-2")
            new_entry.save()
            return HttpResponse("success")
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")





# used in: upload_data3 (the create-object-type modal)
@login_required
def save_edited_object_type(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            object_facts = request_body['li_attr']['attribute_values']
            request_body['li_attr']['attribute_values'] = get_from_db.convert_fact_values_to_the_right_format(object_facts)
            edited_object_type = Object_types.objects.get(id=request_body['id'])
            edited_object_type.parent = request_body['parent']
            edited_object_type.name = request_body['text']
            edited_object_type.li_attr = json.dumps(request_body['li_attr'])
            edited_object_type.save()
            return HttpResponse("success")
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")




# used in: create_attribute_modal.html (i.e. upload_data3,4,5)
@login_required
def save_new_attribute(request):
    print('1')
    if request.method == 'POST':
        print('2')
        try:
            print('3')
            request_body = json.loads(request.body)

            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(request_body)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            new_entry = Attribute(name=request_body['name'], 
                                data_type=request_body['data_type'], 
                                expected_valid_period=request_body['expected_valid_period'], 
                                description=request_body['description'], 
                                format_specification=json.dumps(request_body['format_specification']),
                                first_applicable_object_type=request_body['first_applicable_object_type'],
                                first_relation_object_type=request_body['first_relation_object_type'])
            print('4')
            new_entry.save()
            print('5')
            return HttpResponse("success")
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")


# used in: edit_attribute_modal.html (i.e. upload_data3,4,5)
@login_required
def save_changed_attribute(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            attribute_record = Attribute.objects.get(id=request_body['attribute_id'])

            attribute_record.name = request_body['name']
            attribute_record.data_type = request_body['data_type']
            attribute_record.expected_valid_period = request_body['expected_valid_period']
            attribute_record.description = request_body['description']
            attribute_record.format_specification = json.dumps(request_body['format_specification'])
            attribute_record.first_applicable_object_type = request_body['first_applicable_object_type']
            attribute_record.first_relation_object_type=request_body['first_relation_object_type']
            attribute_record.save()
            return HttpResponse("success")
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")




@login_required
def save_rule(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            

            if ('id' in request_body.keys()):
                rule_id = request_body['id']
                rule_record = Rule.objects.get(id=rule_id)
                rule_record.changed_var_attribute_id = request_body['changed_var_attribute_id']
                rule_record.changed_var_data_type = request_body['changed_var_data_type']
                rule_record.condition_text = request_body['condition_text']
                rule_record.condition_exec = request_body['condition_exec']
                rule_record.aggregation_text = request_body['aggregation_text']
                rule_record.aggregation_exec = request_body['aggregation_exec']
                rule_record.effect_text = request_body['effect_text']
                rule_record.effect_exec = request_body['effect_exec']
                rule_record.effect_is_calculation = request_body['effect_is_calculation']
                rule_record.used_attribute_ids = json.dumps(request_body['used_attribute_ids'])
                rule_record.used_parameter_ids = json.dumps(request_body['used_parameter_ids'])
                rule_record.is_conditionless = request_body['is_conditionless']
                rule_record.has_probability_1 = request_body['has_probability_1']
                rule_record.save()

                # reset all likelihood functions associated with this rule
                likelihood_functions = Likelihood_function.objects.filter(rule_id=rule_id)
                for likelihood_function in likelihood_functions:
                    likelihood_function.list_of_probabilities = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    likelihood_function.save()

                rule_parameters = Rule_parameter.objects.filter(rule_id=rule_id)
                for rule_parameter in rule_parameters:
                    likelihood_functions = Likelihood_function.objects.filter(parameter_id=rule_parameter.id)
                    for likelihood_function in likelihood_functions:
                        likelihood_function.list_of_probabilities = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                        likelihood_function.save()


            else:
                new_entry = Rule(changed_var_attribute_id= request_body['changed_var_attribute_id'],
                                changed_var_data_type= request_body['changed_var_data_type'],
                                condition_text= request_body['condition_text'],
                                condition_exec= request_body['condition_exec'],
                                aggregation_text= request_body['aggregation_text'],
                                aggregation_exec= request_body['aggregation_exec'],
                                effect_text= request_body['effect_text'],
                                effect_exec= request_body['effect_exec'],
                                effect_is_calculation= request_body['effect_is_calculation'],
                                used_attribute_ids= json.dumps(request_body['used_attribute_ids']),
                                used_parameter_ids= json.dumps(request_body['used_parameter_ids']),
                                is_conditionless= request_body['is_conditionless'],
                                has_probability_1= request_body['has_probability_1'])

                new_entry.save()

                rule_id = new_entry.id

            return HttpResponse(rule_id)
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")




# used in: edit_model.html  and edit_model__simulate.html
@login_required
def save_changed_simulation(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            model_record = Simulation_model.objects.get(id=request_body['simulation_id'])

            model_record.aborted = False
            
            model_record.objects_dict = json.dumps(request_body['objects_dict'])
            model_record.y_value_attributes = json.dumps(request_body['y_value_attributes'])
            model_record.manually_set_initial_values = json.dumps(request_body['manually_set_initial_values'])
            model_record.sorted_attribute_ids = json.dumps(request_body['sorted_attribute_ids'])
            model_record.object_type_counts = json.dumps(request_body['object_type_counts'])
            model_record.total_object_count = request_body['total_object_count']
            model_record.number_of_additional_object_facts = request_body['number_of_additional_object_facts']
            model_record.simulation_name = request_body['simulation_name']
            model_record.execution_order_id = request_body['execution_order_id']
            model_record.environment_start_time = request_body['environment_start_time']
            model_record.environment_end_time = request_body['environment_end_time']
            model_record.simulation_start_time = request_body['simulation_start_time']
            model_record.simulation_end_time = request_body['simulation_end_time']
            model_record.max_number_of_instances = request_body['max_number_of_instances']

            # model_record.selected_attribute = request_body['selected_attribute']
            if 'is_timeseries_analysis' in request_body:
                model_record.is_timeseries_analysis = request_body['is_timeseries_analysis']
                model_record.nb_of_tested_parameters = request_body['nb_of_tested_parameters']
                model_record.error_threshold = request_body['error_threshold']
                model_record.run_locally = request_body['run_locally']
                model_record.limit_to_populated_y0_columns = request_body['limit_to_populated_y0_columns']
                

            if 'timestep_size' in request_body:
                model_record.timestep_size = request_body['timestep_size']


            model_record.save()

            return HttpResponse("success")
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")











# used in: learn_rule.html
# @login_required
# def save_learned_rule(request):
#     if request.method == 'POST':
#         try:
#             request_body = json.loads(request.body)

#             learned_rule_record = Learned_rule.objects.get(id=request_body['learned_rule_id'])
#             learned_rule_record.object_type_id = request_body['object_type_id']
#             learned_rule_record.object_filter_facts = json.dumps(request_body['object_filter_facts'])
#             learned_rule_record.specified_factors = json.dumps(request_body['specified_factors'])
#             learned_rule_record.save()

#             return HttpResponse("success")
#         except Exception as error:
#             traceback.print_exc()
#             return HttpResponse(str(error))
#     else:
#         return HttpResponse("This must be a POST request.")


# used in: learn_rule.html
@login_required
def save_changed_object_type_icon(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)

            object_type = Object_types.objects.get(id=request_body['object_type_id'])
            object_type.object_type_icon = request_body['object_type_icon']
            object_type.save()

            return HttpResponse("success")
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")




# used in: edit_simulation__simulate (the edit_object_type_modal)
@login_required
def save_rule_parameter(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            simulation_id = request_body['simulation_id']
            execution_order_id = request_body['execution_order_id']
            object_number = request_body['object_number']
            rule_id = request_body['rule_id']
            new_parameter_dict = request_body['new_parameter_dict']
            if ('id' in request_body):
                parameter = Rule_parameter.objects.get(id=request_body['id'])
                parameter.rule_id = new_parameter_dict['rule_id']
                parameter.parameter_name = new_parameter_dict['parameter_name']
                parameter.min_value = new_parameter_dict['min_value']
                parameter.max_value = new_parameter_dict['max_value']
                parameter.save()

                # if the range was changed: reset the parameter's likelihood_functions
                if request_body['parameter_range_change']:
                    Likelihood_function.objects.filter(parameter_id=request_body['id']).delete()
                    list_of_probabilities = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    likelihood_function = Likelihood_function(simulation_id=simulation_id, execution_order_id=execution_order_id, object_number=object_number, parameter_id=request_body['id'], list_of_probabilities=list_of_probabilities, nb_of_simulations=0, nb_of_sim_in_which_rule_was_used=0, nb_of_tested_parameters=0, nb_of_tested_parameters_in_posterior=0)
                    likelihood_function.save()


                return_dict = {'parameter_id': parameter.id, 'is_new': False, 'request_body':request_body}
                return HttpResponse(json.dumps(return_dict))
            else:
                new_parameter = Rule_parameter(rule_id=new_parameter_dict['rule_id'], parameter_name=new_parameter_dict['parameter_name'], min_value=new_parameter_dict['min_value'], max_value=new_parameter_dict['max_value'])
                new_parameter.save()

                # add to used_parameter_ids of parent rule
                parent_rule = Rule.objects.get(id=new_parameter_dict['rule_id'])
                used_parameter_ids = json.loads(parent_rule.used_parameter_ids)
                parent_rule.used_parameter_ids = used_parameter_ids + [new_parameter.id]
                parent_rule.save()
  
                # add uniform likelihood_function
                list_of_probabilities = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                likelihood_function = Likelihood_function(simulation_id=simulation_id, execution_order_id=execution_order_id, object_number=object_number, parameter_id=new_parameter.id, list_of_probabilities=list_of_probabilities, nb_of_simulations=0, nb_of_sim_in_which_rule_was_used=0, nb_of_tested_parameters=0, nb_of_tested_parameters_in_posterior=0)
                likelihood_function.save()

                return_dict = {'parameter_id': new_parameter.id, 'is_new': True, 'request_body':request_body}
                return HttpResponse(json.dumps(return_dict))

        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")



# used in: edit_simulation__simulate (the edit_object_type_modal)
@login_required
def save_likelihood_function(request):
    if request.method == 'POST':
        print('save_likelihood_function1')
        try:
            request_body = json.loads(request.body)
            print('save_likelihood_function2 - ' + str(request.body))
            if ('id' in request_body):
                print('save_likelihood_function3')
                likelihood_function = Likelihood_function.objects.get(id=request_body['id'])
                likelihood_function.simulation_id = request_body['simulation_id']
                likelihood_function.execution_order_id = request_body['execution_order_id']
                likelihood_function.object_number = request_body['object_number']
                likelihood_function.parameter_id = request_body['parameter_id']
                likelihood_function.list_of_probabilities = json.dumps(request_body['list_of_probabilities'])
                likelihood_function.nb_of_simulations = request_body['nb_of_simulations']
                likelihood_function.nb_of_sim_in_which_rule_was_used = request_body['nb_of_sim_in_which_rule_was_used']
                likelihood_function.nb_of_tested_parameters = request_body['nb_of_tested_parameters']
                likelihood_function.nb_of_tested_parameters_in_posterior = request_body['nb_of_tested_parameters_in_posterior']
                likelihood_function.save()
                print('save_likelihood_function4')
                return HttpResponse(str(likelihood_function.id))
            else:
                print('save_likelihood_function5')
                new_likelihood_function = Likelihood_function(simulation_id=request_body['simulation_id'], execution_order_id=request_body['execution_order_id'], object_number=request_body['object_number'], parameter_id=request_body['parameter_id'], list_of_probabilities=json.dumps(request_body['list_of_probabilities']), nb_of_simulations=request_body['nb_of_simulations'], nb_of_sim_in_which_rule_was_used=request_body['nb_of_sim_in_which_rule_was_used'], nb_of_tested_parameters=request_body['nb_of_tested_parameters'], nb_of_tested_parameters_in_posterior=request_body['nb_of_tested_parameters_in_posterior'])
                new_likelihood_function.save()
                print('save_likelihood_function6')
                return HttpResponse(str(new_likelihood_function.id))
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")




# used in: edit_simulation__simulate (the saveSimulation function) and select_execution_order_modal.html
@login_required
def save_changed_execution_order(request):
    print('-------  save_changed_execution_order  -----------')
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            if ('id' in request_body):
                execution_order_record = Execution_order.objects.get(id=request_body['id'])
                if ('name' in request_body):
                    execution_order_record.name = request_body['name']
                if ('description' in request_body):
                    execution_order_record.description = request_body['description']
                if ('execution_order' in request_body):
                    execution_order_record.execution_order = json.dumps(request_body['execution_order'])
                execution_order_record.save()
                return HttpResponse(str(execution_order_record.id))
            else:
                execution_order_record = Execution_order(name=name,description=request_body['description'], execution_order=json.dumps(request_body['execution_order']))
                execution_order_record.save()
                return HttpResponse(str(execution_order_record.id))
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")


# used in: select_execution_order_modal.html
@login_required
def save_new_execution_order(request):
    if request.method == 'POST':

        request_body = json.loads(request.body)
        id_to_copy = request_body['id_to_copy']
        name = request_body['name']
        description = request_body['description']

        copied_execution_order = Execution_order.objects.get(id=id_to_copy).execution_order

        new_execution_order_record = Execution_order(name=name,
                                            description=description,
                                            execution_order=copied_execution_order)

        new_execution_order_record.save()
        new_execution_order_id = new_execution_order_record.id
        new_execution_order_dict = {'id':new_execution_order_id, 'name':name, 'description':description}
        return HttpResponse(json.dumps(new_execution_order_dict))

    else:
        return HttpResponse("This must be a POST request.")





# ==================
# DELETE
# ==================

# used in: upload_data3 (the create-object-type modal)
@login_required
def delete_object_type(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            object_id = request_body['object_id']

            delted_object = Object_types.objects.get(id=object_id)

            children_of_deleted_object  = Object_types.objects.filter(parent=object_id)
            for child in children_of_deleted_object:
                child.parent = delted_object.parent
                child.save()

            delted_object.delete()
            return HttpResponse("success")
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")




# used in: edit_attribute_modal.html
@login_required
def delete_attribute(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            attribute_id = request_body['attribute_id']

            attribute = Attribute.objects.get(id=attribute_id)
            attribute.delete()
            return HttpResponse("success")
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")


# used in: edit_simulation__simulate.html
@login_required
def delete_rule(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            rule_id = request_body['rule_id']
            rule = Rule.objects.get(id=rule_id)
            rule.delete()

            likelihood_functions = Likelihood_function.objects.filter(rule_id=rule_id)
            likelihood_functions.delete()

            rule_parameters = Rule_parameter.objects.filter(rule_id=rule_id)
            rule_parameters.delete()
            return HttpResponse("success")

        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")



# used in: edit_simulation__simulate.html
@login_required
def delete_parameter(request):
    if request.method == 'POST':
        try:
            request_body = json.loads(request.body)
            parameter_id = request_body['parameter_id']
            parameter = Rule_parameter.objects.get(id=parameter_id)
            parameter.delete()

            likelihood_functions = Likelihood_function.objects.filter(parameter_id=parameter_id)
            likelihood_functions.delete()
            return HttpResponse("success")
            
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")



@login_required
def delete_execution_order(request):
    if request.method == 'POST':
        try:
            execution_order_id = int(request.body)
            parameter = Execution_order.objects.get(id=execution_order_id)
            parameter.delete()
            return HttpResponse("success")
            
        except Exception as error:
            traceback.print_exc()
            return HttpResponse(str(error))
    else:
        return HttpResponse("This must be a POST request.")


# ==================
# PROCESS
# ==================

# used in upload_data5.html
@login_required
def edit_column(request): 
    request_body = json.loads(request.body)

    transformation = request_body['transformation']
    transformation = transformation.replace('"','')

    subset_specification = request_body['subset_specification']
    subset_specification = subset_specification.replace('"','')
    if subset_specification.replace(' ','') != "":
        subset_specification = subset_specification + " and "

    column = request_body['edited_column']
    edited_column = column
    errors = []

    try:
        entire_code = "edited_column = [" + transformation + " if " + subset_specification + "value is not None else value for value in " + str(column) + " ]"
        execution_results = {}
        exec(entire_code, globals(), execution_results)
        edited_column = execution_results['edited_column']
    except Exception as error:
        traceback.print_exc()
        errors.append(str(error))

    response = {}
    response['errors'] = errors
    response['edited_column'] = edited_column

    return HttpResponse(json.dumps(response))



# used in learn_rule.html
# @login_required
# def learn_rule_from_factors(request):
#     learned_rule_id = int(request.GET.get('learned_rule_id', 0))
#     the_rule_learner = rule_learner.Rule_Learner(learned_rule_id)
#     response = the_rule_learner.run()

#     return HttpResponse(json.dumps(response))



# ==================
# COLUMN FORMAT
# ==================

@login_required
def suggest_attribute_format(request): 
    request_body = json.loads(request.body)
    column_values = request_body['column_values']
    column_dict = {'column': column_values}
    constraints_dict = tdda_functions.suggest_attribute_format(column_dict)
    return HttpResponse(json.dumps(constraints_dict))



@login_required
def get_columns_format_violations(request):
    request_body = json.loads(request.body)
    attribute_id = request_body['attribute_id']
    column_values = request_body['column_values']
    violating_rows = tdda_functions.get_columns_format_violations(attribute_id, column_values)
    return HttpResponse(json.dumps(violating_rows))






def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def is_float(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False


@login_required
def check_single_fact_format(request):
    attribute_id = request.GET.get('attribute_id', '')
    operator = request.GET.get('operator', '')
    value = request.GET.get('value', '')
    print('-----------------------------------------------------------')
    print(value)

    response = {}
    response['fact_number'] = int(request.GET.get('fact_number', ''))

    if (attribute_id == '') or (value == '') or not is_int(attribute_id):
        response['format_violation_text'] = ''
        return HttpResponse(json.dumps(response))
        
    attribute_id = int(attribute_id)
    attribute_record = Attribute.objects.get(id=attribute_id)

    if attribute_record is None:
        response['format_violation_text'] = 'This attribute wasn''t found.'
        return HttpResponse(json.dumps(response))
    
    attribute_name = attribute_record.name
    format_specification = json.loads(attribute_record.format_specification)
    if (operator not in ['=', '>', '<', 'in']):
        response['format_violation_text'] = '"' + operator +'" is not a valid operator.'
        return HttpResponse(json.dumps(response))

    # if (operator == 'in') and ('allowed_values' not in format_specification['fields']['column'].keys()):
    #     response['format_violation_text'] = 'The "in" operator is only permitted for categorical attributes.'
    #     return HttpResponse(json.dumps(response))
        
    if (operator in ['>', '<']) and (format_specification['fields']['column']['type'] not in ['int','real']):
        response['format_violation_text'] = 'The "<" and ">" operators are only permitted for attributes with type "real" or "int".'
        return HttpResponse(json.dumps(response))

    if format_specification['fields']['column']['type']=='int':
        if is_int(value):
            value = int(value)
        else:
            response['format_violation_text'] = attribute_name +'-values must be integers. ' + value + ' is not an integer.'
            return HttpResponse(json.dumps(response))

    if format_specification['fields']['column']['type']=='real':
        if is_float(value):
            value = float(value)
        else:
            response['format_violation_text'] = attribute_name +'-values must be real numbers. ' + value + ' is not a number.'
            return HttpResponse(json.dumps(response))

    if format_specification['fields']['column']['type']=='bool':
        if value.lower() in ['true', 'true ', 'tru', 't', 'yes', 'yes ', 'y']:
            value = True
        elif value.lower() in ['false', 'false ', 'flase', 'f', 'no', 'no ', 'n']:
            value = False
        else:
            response['format_violation_text'] = attribute_name +'-values must be "true" or "false", not ' + value + '.'
            return HttpResponse(json.dumps(response))

    if operator == '=':
        violating_columns = tdda_functions.get_columns_format_violations(attribute_id, [value])
        if len(violating_columns) > 0:
            format_violation_text = 'The value "' + str(value) + '" does not satisfy the required format for ' + attribute_name + '-values. <br />'
            format_violation_text += 'It must satisfy: <ul>'
            for key in format_specification['fields']['column'].keys():
                format_violation_text += '<li>' + str(key) + ' = ' + str(format_specification['fields']['column'][key]) + '</li>'
            format_violation_text += '</ul>'
            response['format_violation_text'] = format_violation_text
            return HttpResponse(json.dumps(response))

    if operator == 'in':
        list_of_values = json.loads(value)
        for individual_value in list_of_values:
            violating_columns = tdda_functions.get_columns_format_violations(attribute_id, [individual_value])
            if len(violating_columns) > 0:
                format_violation_text = 'The value "' + str(individual_value) + '" does not satisfy the required format for ' + attribute_name + '-values. <br />'
                format_violation_text += 'It must satisfy: <ul>'
                for key in format_specification['fields']['column'].keys():
                    format_violation_text += '<li>' + str(key) + ' = ' + str(format_specification['fields']['column'][key]) + '</li>'
                format_violation_text += '</ul>'
                response['format_violation_text'] = format_violation_text
                return HttpResponse(json.dumps(response))


    response['format_violation_text'] = ''
    return HttpResponse(json.dumps(response))

    



 # ===============================================================
 #  __  __           _      _     
 # |  \/  |         | |    | |    
 # | \  / | ___   __| | ___| |___ 
 # | |\/| |/ _ \ / _` |/ _ \ / __|
 # | |  | | (_) | (_| |  __/ \__ \
 # |_|  |_|\___/ \__,_|\___|_|___/
 #
 # ===============================================================


@login_required
def execution_order_scores(request):
    return render(request, 'tool/execution_order_scores.html')




@login_required
def get_execution_order_scores(request):
    print('------------------ get_execution_order_scores --------------------------------')
    from django.db import connection
    import boto3
    response = {'simulations': [], 'scores': {}}

    sql_string = '''
        SELECT sub_query.simulation_id, 
               sub_query.execution_order_id, 
               AVG(nb_of_simulations) as nb_of_simulations,
               AVG(nb_of_tested_parameters) as nb_of_tested_parameters,
               AVG(sub_query.nb_of_tested_parameters_in_posterior) as nb_of_tested_parameters_in_posterior
        FROM ( 
                SELECT  simulation_id, 
                        execution_order_id, 
                        rule_id,
                        parameter_id,
                        nb_of_simulations,
                        nb_of_tested_parameters,
                        nb_of_tested_parameters_in_posterior,
                        ROW_NUMBER() OVER(PARTITION BY simulation_id, execution_order_id, rule_id, parameter_id  ORDER BY id DESC) AS rank
                FROM collection_likelihood_function 
            ) as sub_query
        WHERE sub_query.rank = 1
        GROUP BY sub_query.simulation_id, sub_query.execution_order_id
    ''' 

    run_simulations_df = pd.read_sql_query(sql_string, connection)

    # add execution_orders
    all_execution_order_ids = list(run_simulations_df['execution_order_id'].unique())
    response['execution_orders'] = {execution_order.id: {'name':execution_order.name, 'sum_of_scores':0, 'total_nb_of_validation_datapoints':0} for execution_order in Execution_order.objects.filter(id__in=all_execution_order_ids).order_by('id')}


    # add simuations
    all_simuation_ids = list(run_simulations_df['simulation_id'].unique())
    simulation_models = Simulation_model.objects.filter(id__in=all_simuation_ids).order_by('id')
    for simulation_model in simulation_models:

            
        s3 = boto3.resource('s3')
        obj = s3.Object('elasticbeanstalk-eu-central-1-662304246363', 'SimulationModels/simulation_' + str(simulation_model.id) + '_validation_data.json')
        y0_values = json.loads(obj.get()['Body'].read().decode('utf-8'))['y0_values']
        y0_values = np.asarray(y0_values, dtype=object, order='c').squeeze()
        y0_values = np.atleast_1d(y0_values)
        y0_values_df = pd.DataFrame(list(y0_values))
        y0_values_df = y0_values_df.fillna(np.nan)
        validation_columns = [col for col in y0_values_df.columns if 'period' in col and 'period0' not in col]
        y0_values_df = y0_values_df[validation_columns]
        number_of_validation_datapoints = int(np.sum(y0_values_df.count()))


        response['simulations'].append({'simulation_id': simulation_model.id, 'simulation_name': simulation_model.simulation_name, 'number_of_validation_datapoints': number_of_validation_datapoints}) 
        response['scores'][simulation_model.id] = {}

        for index, row in run_simulations_df[run_simulations_df['simulation_id']==simulation_model.id].iterrows():
            if row['nb_of_simulations'] > 0:

                execution_order_id = int(row['execution_order_id'])
                learn_parameters_result = Learn_parameters_result.objects.filter(simulation_id=simulation_model.id, execution_order_id=execution_order_id).order_by('-id').first()
                all_priors_df = pd.DataFrame.from_dict(json.loads(learn_parameters_result.all_priors_df), orient='index')
                if len(all_priors_df) > row['nb_of_tested_parameters_in_posterior']:
                    all_priors_df.index = range(len(all_priors_df))
                    score = 1 - all_priors_df.loc[:row['nb_of_tested_parameters_in_posterior'], 'error'].mean()

                    response['scores'][simulation_model.id][execution_order_id] = {'score': score, 'number_of_validation_datapoints': number_of_validation_datapoints}
                    response['execution_orders'][execution_order_id]['sum_of_scores'] += score*number_of_validation_datapoints
                    response['execution_orders'][execution_order_id]['total_nb_of_validation_datapoints'] += number_of_validation_datapoints

    for execution_order_id in response['execution_orders'].keys():
        if response['execution_orders'][execution_order_id]['total_nb_of_validation_datapoints'] > 0:
            response['execution_orders'][execution_order_id]['overall_score'] = response['execution_orders'][execution_order_id]['sum_of_scores']/response['execution_orders'][execution_order_id]['total_nb_of_validation_datapoints']
    

    print(json.dumps(response['simulations']))
    print(json.dumps(response['scores']))
    print(json.dumps(response['execution_orders']))
    return HttpResponse(json.dumps(response).replace(': NaN', ': null'))


 # ===============================================================
 #   ____                          _____        _        
 #  / __ \                        |  __ \      | |       
 # | |  | |_   _  ___ _ __ _   _  | |  | | __ _| |_ __ _ 
 # | |  | | | | |/ _ \ '__| | | | | |  | |/ _` | __/ _` |
 # | |__| | |_| |  __/ |  | |_| | | |__| | (_| | || (_| |
 #  \___\_\\__,_|\___|_|   \__, | |_____/ \__,_|\__\__,_|
 #                          __/ |                        
 #                         |___/                         
 # ===============================================================



@login_required
def query_data(request):
    object_hierachy_tree = get_from_db.get_object_hierachy_tree()
    return render(request, 'tool/query_data.html',{'object_hierachy_tree':object_hierachy_tree})


def download_file1(request):
    displayed_table_dict = json.loads(request.body)
    displayed_table_df = pd.DataFrame(displayed_table_dict)
    current_timestamp = int(round(time.time() * 1000))
    filename = str(current_timestamp) + ".csv"
    displayed_table_df.to_csv("collection/static/webservice files/downloaded_data_files/" + filename)
    return HttpResponse(current_timestamp)

def download_file2(request, file_name, file_type):
    # filename = request.GET.get('filename', '')
    displayed_table_df = pd.read_csv("collection/static/webservice files/downloaded_data_files/" + file_name + ".csv")
    column_names = displayed_table_df.columns
    
    if file_type=='xls':
        response = HttpResponse(content_type='application/ms-excel')
        response['Content-Disposition'] = 'attachment; filename="tree_of_knowledge_data.xls"'

        #creating workbook
        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet("sheet1")

        # headers
        font_style = xlwt.XFStyle()
        font_style.font.bold = True
        for column_number, column_name in enumerate(column_names[1:]):
            ws.write(0, column_number, column_name, font_style)

        # table body
        font_style.font.bold = False
        for index, row in displayed_table_df.iterrows():
            for column_number, value in enumerate(row.tolist()[1:]):
                ws.write(index + 1, column_number, value, font_style)

        wb.save(response)
        return response

    elif file_type=='csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="tree_of_knowledge_data.csv"'

        writer = csv.writer(response, csv.excel)
        response.write(u'\ufeff'.encode('utf8'))

        # headers
        smart_str_column_names = [smart_str(name) for name in column_names[1:]]
        writer.writerow(smart_str_column_names)

        # table body
        for index, row in displayed_table_df.iterrows():
            smart_str_row = [smart_str(value) for value in row.tolist()[1:]]
            writer.writerow(smart_str_row)
            
        return response




def download_file2_csv(request):
    filename = request.GET.get('filename', '')
    displayed_table_dict = pd.read_csv("collection/static/webservice files/downloaded_data_files/" + filename + ".csv")
    
    # Create the HttpResponse object with the appropriate CSV header.
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="somefilename.csv"'

    writer = csv.writer(response)
    writer.writerow(displayed_table_df.columns)
    for index, row in displayed_table_df.iterrows():
        writer.writerow(row.tolist())
    return response




 # ==========================================================================
 #   _____ _                 _       _   _             
 #  / ____(_)               | |     | | (_)            
 # | (___  _ _ __ ___  _   _| | __ _| |_ _  ___  _ __  
 #  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \ 
 #  ____) | | | | | | | |_| | | (_| | |_| | (_) | | | |
 # |_____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|
 # 
 # ==========================================================================

@login_required
def edit_simulation_new(request):    
    simulation_model = Simulation_model(aborted=False,
                                        run_number=0,
                                        user=request.user, 
                                        is_timeseries_analysis=True,
                                        objects_dict='{}', 
                                        y_value_attributes='[]', 
                                        manually_set_initial_values = '{}',
                                        sorted_attribute_ids='[]', 
                                        object_type_counts='{}',
                                        total_object_count=0,
                                        number_of_additional_object_facts=2,
                                        simulation_name='New Simulation',
                                        execution_order_id=1,
                                        not_used_rules='{}',
                                        environment_start_time=946684800, 
                                        environment_end_time=1577836800, 
                                        simulation_start_time=946684800, 
                                        simulation_end_time=1577836800, 
                                        timestep_size=31622400,
                                        nb_of_tested_parameters=40,
                                        max_number_of_instances=2000,
                                        error_threshold=0.2,
                                        run_locally=False,
                                        limit_to_populated_y0_columns=False,
                                        data_querying_info='{"timestamps":{}, "table_sizes":{}, "relation_sizes":{}}',
                                        all_priors_df='{}',)
    simulation_model.save()
    new_simulation_id = simulation_model.id
    return redirect('edit_simulation', simulation_id=new_simulation_id)



@login_required
def edit_simulation(request, simulation_id):
    simulation_model = Simulation_model.objects.get(id=simulation_id)
    if request.method == 'POST':
        the_simulator = simulation.Simulator(simulation_id, False)
        print('learning simulation')
        parameters_were_learned = the_simulator.learn_and_run_best_parameter()
        print('simulation learned')
        return HttpResponse(json.dumps(parameters_were_learned))
        # print('simulation completed, redirecting..')
        # if 'DB_CONNECTION_URL' in os.environ:
        #     return redirect('https://www.treeofknowledge.ai/tool/analyse_simulation', simulation_id=simulation_id)
        # else:
        #     return redirect('analyse_simulation', simulation_id=simulation_id)


    simulation_results_exist = Learn_parameters_result.objects.filter(simulation_id=simulation_id, execution_order_id=simulation_model.execution_order_id).count() > 0
    available_object_types = get_from_db.get_most_commonly_used_object_types()
    object_icons = [icon_name[:-4] for icon_name in os.listdir("collection/static/images/object_icons/")]
    available_relations = get_from_db.get_available_relations()
    available_execution_orders = get_from_db.get_available_execution_orders()
    return render(request, 'tool/edit_simulation.html', {'simulation_model':simulation_model, 'available_object_types': available_object_types, 'object_icons': object_icons, 'available_relations':available_relations, 'available_execution_orders':available_execution_orders, 'simulation_results_exist': simulation_results_exist})









@login_required
def copy_simulation(request):
    simulation_id = int(request.GET.get('simulation_id', ''))
    simulation_record = Simulation_model.objects.get(id=simulation_id)
    simulation_record.id = None
    simulation_record.simulation_name = 'Copy of: ' + simulation_record.simulation_name
    simulation_record.save()
    id_of_new_simulation_record = simulation_record.id
    return HttpResponse(str(id_of_new_simulation_record))




@login_required
def get_simulation_progress(request):
    simulation_id = request.GET.get('simulation_id', '')

    with open('collection/static/webservice files/runtime_data/simulation_progress_' + simulation_id + '.txt') as file:       
        progress = file.readline()

    return HttpResponse(progress)



@login_required
def analyse_learned_parameters(request, simulation_id, execution_order_id):
    print('analyse_learned_parameters')
   
    with open('collection/static/webservice files/runtime_data/simulation_progress_' + str(simulation_id) + '.txt', "w") as progress_tracking_file:
        progress_tracking_file.write(json.dumps({"learning_likelihoods": False, "nb_of_accepted_simulations_total": "", "nb_of_accepted_simulations_current": "" , "running_monte_carlo": False, "monte_carlo__simulation_number": "", "monte_carlo__number_of_simulations":  "",}))
    simulation_model = Simulation_model.objects.get(id=simulation_id)
    learn_parameters_result = Learn_parameters_result.objects.filter(simulation_id=simulation_model.id, execution_order_id=execution_order_id).order_by('-id').first()
    available_execution_orders = get_from_db.get_available_execution_orders()
    execution_order = Execution_order.objects.get(id=execution_order_id)
    results_from_all_runs = Learn_parameters_result.objects.filter(simulation_id=simulation_model.id, execution_order_id=execution_order_id).order_by('-run_number')
    return render(request, 'tool/analyse_learned_parameters.html', {'simulation_model':simulation_model, 'learn_parameters_result': learn_parameters_result, 'available_execution_orders':available_execution_orders, 'execution_order':execution_order, 'results_from_all_runs':results_from_all_runs})



@login_required
def analyse_new_simulation(request, simulation_id, execution_order_id, parameter_number):
    print('analyse_simulation')

    with open('collection/static/webservice files/runtime_data/simulation_progress_' + str(simulation_id) + '.txt', "w") as progress_tracking_file:
        progress_tracking_file.write(json.dumps({"learning_likelihoods": False, "nb_of_accepted_simulations_total": "", "nb_of_accepted_simulations_current": "" , "running_monte_carlo": False, "monte_carlo__simulation_number": "", "monte_carlo__number_of_simulations":  "",}))
    simulation_model = Simulation_model.objects.get(id=simulation_id)
    print('simulation_id=%s, parameter_number=%s'  % (simulation_id, parameter_number))
    learn_parameters_result = Learn_parameters_result.objects.filter(simulation_id=simulation_model.id, execution_order_id=execution_order_id).order_by('-id').first()
    monte_carlo_result = Monte_carlo_result.objects.filter(simulation_id=simulation_id, execution_order_id=execution_order_id, parameter_number=parameter_number, is_new_parameter=True).order_by('-id').first()
    execution_order = Execution_order.objects.get(id=execution_order_id)
    return render(request, 'tool/analyse_simulation.html', {'simulation_model':simulation_model, 'learn_parameters_result': learn_parameters_result, 'monte_carlo_result':monte_carlo_result, 'execution_order':execution_order})



@login_required
def analyse_simulation(request, simulation_id, execution_order_id, parameter_number):
    print('analyse_simulation')

    with open('collection/static/webservice files/runtime_data/simulation_progress_' + str(simulation_id) + '.txt', "w") as progress_tracking_file:
        progress_tracking_file.write(json.dumps({"learning_likelihoods": False, "nb_of_accepted_simulations_total": "", "nb_of_accepted_simulations_current": "" , "running_monte_carlo": False, "monte_carlo__simulation_number": "", "monte_carlo__number_of_simulations":  "",}))
    simulation_model = Simulation_model.objects.get(id=simulation_id)
    print('simulation_id=%s, parameter_number=%s'  % (simulation_id, parameter_number))
    learn_parameters_result = Learn_parameters_result.objects.filter(simulation_id=simulation_model.id, execution_order_id=execution_order_id).order_by('-id').first()
    monte_carlo_result = Monte_carlo_result.objects.filter(simulation_id=simulation_id, execution_order_id=execution_order_id, parameter_number=parameter_number, is_new_parameter=False).order_by('-id').first()
    execution_order = Execution_order.objects.get(id=execution_order_id)
    return render(request, 'tool/analyse_simulation.html', {'simulation_model':simulation_model, 'learn_parameters_result': learn_parameters_result, 'monte_carlo_result':monte_carlo_result, 'execution_order':execution_order})


@login_required
def abort_simulation(request):
    simulation_id = int(request.GET.get('simulation_id', ''))
    model_record = Simulation_model.objects.get(id=simulation_id)
    model_record.aborted = True
    model_record.save()
    return HttpResponse("success")


@login_required
def run_single_monte_carlo(request):
    print('----- run_single_monte_carlo -----')
    if request.method == 'POST':
        print('rsmc 1')
        request_body = json.loads(request.body)
        print('rsmc 2')
        the_simulator = simulation.Simulator(request_body['simulation_id'], True)
        print('rsmc 3')
        parameter_number = the_simulator.run_single_monte_carlo(request_body['number_of_entities_to_simulate'], request_body['prior_dict'], request_body['parameter_number'])
        print('rsmc 4')

    return HttpResponse(parameter_number)



# @login_required
# def setup_rule_learning(request, simulation_id):

#     simulation_model = Simulation_model.objects.get(id=simulation_id)
#     objects_dict = json.loads(simulation_model.objects_dict)

#     object_number = int(request.POST.get('object_number', None))
#     attribute_id = int(request.POST.get('attribute_id', None))

#     valid_times = []
#     times = np.arange(simulation_model.simulation_start_time + simulation_model.timestep_size, simulation_model.simulation_end_time, simulation_model.timestep_size)
#     for index in range(len(times)):
#         valid_times.append([int(times[index]), int(times[index + 1])])

#     learned_rule = Learned_rule(object_type_id=objects_dict[str(object_number)]['object_type_id'], 
#                                 object_type_name=objects_dict[str(object_number)]['object_type_name'],
#                                 attribute_id=attribute_id,
#                                 attribute_name=objects_dict[str(object_number)]['object_attributes'][str(attribute_id)]['attribute_name'],
#                                 object_filter_facts=json.dumps(objects_dict[str(object_number)]['object_filter_facts']),
#                                 specified_factors = '{}',
#                                 sorted_factor_numbers = '[]',
#                                 valid_times= json.dumps(valid_times),
#                                 min_score_contribution = 0.01,
#                                 max_p_value = 0.05,
#                                 user=request.user)

#     learned_rule.save()
#     learned_rule_id = learned_rule.id



#     return redirect('learn_rule', learned_rule_id=learned_rule_id)



# @login_required
# def learn_rule(request, learned_rule_id):
#     learned_rule = Learned_rule.objects.get(id=learned_rule_id)

#     available_attributes = []
#     list_of_parent_objects = get_from_db.get_list_of_parent_objects(learned_rule.object_type_id)
#     for parent_object in list_of_parent_objects:
#         available_attributes.extend(list(Attribute.objects.filter(first_applicable_object_type=parent_object['id']).values('name', 'id', 'data_type')))

#     return render(request, 'tool/learn_rule.html', {'learned_rule':learned_rule, 'available_attributes': available_attributes})




 # ==========================================================================
 #
 #              _           _         _____                      
 #     /\      | |         (_)       |  __ \                     
 #    /  \   __| |_ __ ___  _ _ __   | |__) |_ _  __ _  ___  ___ 
 #   / /\ \ / _` | '_ ` _ \| | '_ \  |  ___/ _` |/ _` |/ _ \/ __|
 #  / ____ \ (_| | | | | | | | | | | | |  | (_| | (_| |  __/\__ \
 # /_/    \_\__,_|_| |_| |_|_|_| |_| |_|   \__,_|\__, |\___||___/
 #                                                __/ |          
 #                                               |___/    
 # 
 # ==========================================================================

@staff_member_required
def admin_page(request):
    number_of_users = User.objects.count()
    number_of_newsletter_subscribers = Newsletter_subscriber.objects.count()
    return render(request, 'admin/admin_page.html', {'number_of_users': number_of_users, 'number_of_newsletter_subscribers': number_of_newsletter_subscribers})


# ==================
# INSPECT
# ==================

@staff_member_required
def inspect_object(request, object_id):
    return render(request, 'admin/inspect_object.html', {'object_id': object_id})

@staff_member_required
def get_object(request):
    object_id = request.GET.get('object_id', '')
    individual_object = admin_fuctions.inspect_individual_object(object_id)
    return HttpResponse(json.dumps(individual_object))

@staff_member_required
def inspect_upload(request, upload_id):
    return render(request, 'admin/inspect_upload.html', {'upload_id': upload_id})

@staff_member_required
def get_uploaded_dataset(request):
    upload_id = request.GET.get('upload_id', None)
    uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id)
    uploaded_dataset_dict = {   'data_table_json':json.loads(uploaded_dataset.data_table_json),
                                'object_id_column':json.loads(uploaded_dataset.object_id_column),
                                'data_source':uploaded_dataset.data_source,
                                'data_generation_date':uploaded_dataset.data_generation_date.strftime('%Y-%m-%d %H:%M'),
                                'correctness_of_data':uploaded_dataset.correctness_of_data,
                                'object_type_name':uploaded_dataset.object_type_name,
                                'meta_data_facts':uploaded_dataset.meta_data_facts}

    return HttpResponse(json.dumps(uploaded_dataset_dict))



@staff_member_required
def inspect_logged_variables(request):
    return render(request, 'admin/inspect_logged_variables.html')


@staff_member_required
def get_logged_variables_last_x_minutes(request):
    last_x_minutes = int(request.GET.get('last_x_minutes', None))
    earliest_time = time.time() - last_x_minutes*60
    logged_variable_records = Logged_variable.objects.filter(logged_time__gt=earliest_time).order_by('logged_time') 

    response = []
    for logged_variable_record in logged_variable_records:
        logged_datetime = datetime.fromtimestamp(logged_variable_record.logged_time)
        time_string = '%02d:%02d:%s' % (logged_datetime.hour, logged_datetime.minute, logged_datetime.second)
        response.append({'logged_time': logged_variable_record.logged_time, 'time_string':time_string, 'variable_name': logged_variable_record.variable_name, 'variable_value': logged_variable_record.variable_value})
    return HttpResponse(json.dumps(response))




@staff_member_required
def run_query(request):
    return render(request, 'admin/run_query.html')


@staff_member_required
def get_query_results(request):
    from django.db import connection
    if request.method == 'POST':
        query = request.body.decode('utf-8')
        print('-------------------------------')
        print(query)
        print('-------------------------------')
        if 'select' in query.lower():
            query_result_df = pd.read_sql_query(query, connection)
            query_result_df = query_result_df.where(pd.notnull(query_result_df), None)
            return_dict = { 'table_headers':list(query_result_df.columns), 
                            'table_data': query_result_df.values.tolist()}
            return HttpResponse(json.dumps(return_dict))
        else:
            with connection.cursor() as cursor:
                cursor.execute(query)
            return HttpResponse('success')

    

# ==================
# SHOW
# ==================

@staff_member_required
def show_attributes(request):
    attributes = Attribute.objects.all().order_by('id')
    return render(request, 'admin/show_attributes.html', {'attributes': attributes,})

@staff_member_required
def show_object_types(request):
    object_types = Object_types.objects.all()
    return render(request, 'admin/show_object_types.html', {'object_types': object_types,})



@staff_member_required
def show_newsletter_subscribers(request):
    newsletter_subscribers = Newsletter_subscriber.objects.all().order_by('email')
    return render(request, 'admin/show_newsletter_subscribers.html', {'newsletter_subscribers': newsletter_subscribers,})

@staff_member_required
def show_users(request):
    users = User.objects.all()
    return render(request, 'admin/show_users.html', {'users': users,})



# ==================
# DATA CLEANING
# ==================

@staff_member_required
def possibly_duplicate_objects_without_keys(request):
    object_types = list(Object_types.objects.all().values())
    object_types_names = {str(object_type['id']):object_type['name'] for object_type in object_types}
    return render(request, 'admin/possibly_duplicate_objects_without_keys.html', {'object_types_names':object_types_names})


@staff_member_required
def find_possibly_duplicate_objects_without_keys(request):
    print('find_possibly_duplicate_objects_without_keys')
    admin_fuctions.find_possibly_duplicate_objects_without_keys()


@staff_member_required
def get_possibly_duplicate_objects_without_keys(request):
    with open('collection/static/webservice files/runtime_data/duplicate_objects_by_object_type.txt', 'r') as file:
        duplicate_objects_by_object_type = file.read().replace('\n', '')
    return HttpResponse(duplicate_objects_by_object_type)

# ----

@staff_member_required
def possibly_duplicate_objects_with_keys(request):
    object_types = list(Object_types.objects.all().values())
    object_types_names = {str(object_type['id']):object_type['name'] for object_type in object_types}
    return render(request, 'admin/possibly_duplicate_objects_with_keys.html', {'object_types_names':object_types_names})



@staff_member_required
def get_possibly_duplicate_objects_with_keys(request):
    object_type_id = request.GET.get('object_type_id', '')
    key_attribute_id = request.GET.get('key_attribute_id', '')
    possibly_duplicate_objects = admin_fuctions.get_possibly_duplicate_objects_with_keys(object_type_id, key_attribute_id)
    return HttpResponse(json.dumps(possibly_duplicate_objects))


# ----
@staff_member_required
def delete_objects_page(request):
    return render(request, 'admin/delete_objects.html')


@staff_member_required
def delete_objects(request):
    if request.method == 'POST':
        print('===================')
        print(request.body)
        object_ids = json.loads(request.body)
        print('+++++++++++++++++++++++')
        print(str(object_ids))
        Object.objects.filter(id__in=object_ids).delete()
        Data_point.objects.filter(object_id__in=object_ids).delete()
    return HttpResponse('Objects deleted!')



# ----
@staff_member_required
def delete_upload_page(request):
    return render(request, 'admin/delete_upload.html')


@staff_member_required
def delete_upload(request):
    upload_id = int(request.GET.get('upload_id', ''))
    print('+++++++++++++++++++++++')
    print(str(upload_id))
    Uploaded_dataset.objects.get(id=upload_id).delete()
    Data_point.objects.filter(upload_id=upload_id).delete()
    return HttpResponse('Upload deleted!')

# ----
@staff_member_required
def delete_simulation_page(request):
    return render(request, 'admin/delete_simulation.html')


@staff_member_required
def delete_simulation(request):
    simulation_id = int(request.GET.get('simulation_id', ''))
    print('+++++++++++++++++++++++')
    print(str(simulation_id))
    Simulation_model.objects.get(id=simulation_id).delete()
    Learn_parameters_result.objects.filter(simulation_id=simulation_id).delete()
    Monte_carlo_result.objects.filter(simulation_id=simulation_id).delete()
    Likelihood_function.objects.filter(simulation_id=simulation_id).delete()
    return HttpResponse('Simulation deleted!')


# ==================
# MODEL FIXES
# ==================


@login_required
def salvage_cancelled_simulation_page(request):
    return render(request, 'admin/salvage_cancelled_simulation.html')

@login_required
def salvage_cancelled_simulation(request, simulation_id, run_number):
    
    if request.method == 'POST':
        simulation_id = int(simulation_id)
        run_number = int(run_number)
        simulation_model = Simulation_model.objects.get(id=simulation_id)
        the_simulator = simulation.Simulator(simulation_id, False)
        success = the_simulator.salvage_cancelled_simulation(run_number)
        return HttpResponse(json.dumps(success))


@login_required
def show_validation_data(request):
    return render(request, 'admin/show_validation_data.html')

@login_required
def get_validation_data(request):
    import boto3
    simulation_id = int(request.GET.get('simulation_id', ''))
    s3 = boto3.resource('s3')
    obj = s3.Object('elasticbeanstalk-eu-central-1-662304246363', 'SimulationModels/simulation_' + str(simulation_id) + '_validation_data.json')
    validation_data = obj.get()['Body'].read().decode('utf-8')
    return HttpResponse(validation_data.replace('NaN', 'null'))
    

# ==================
# VARIOUS SCRIPTS
# ==================

@staff_member_required
def various_scripts(request):
    return render(request, 'admin/various_scripts.html',)


@staff_member_required
def remove_null_datapoints(request):
    print('views.remove_null_datapoints')
    admin_fuctions.remove_null_datapoints()
    return HttpResponse('success')

@staff_member_required
def remove_duplicate_datapoints(request):
    admin_fuctions.remove_duplicates()
    return HttpResponse('success')


@staff_member_required
def backup_database(request):
    success_for_object_types = admin_fuctions.backup_object_types()
    success_for_attributes = admin_fuctions.backup_attributes()

    if (success_for_object_types and success_for_attributes):
        return HttpResponse('success')
    else:
        return HttpResponse('An error occured')


@staff_member_required
def close_db_connections(request):
    from django import db
    db.connections.close_all()
    return HttpResponse('success')



@staff_member_required
def clear_database(request):
    admin_fuctions.clear_object_types()
    admin_fuctions.clear_attributes()
    return HttpResponse('done')     


@staff_member_required
def populate_database(request):
    admin_fuctions.populate_object_types()
    admin_fuctions.populate_attributes()
    return HttpResponse('done')




@staff_member_required
def upload_file(request):
    errors = []
    if request.method == 'POST':
        form1 = UploadFileForm(request.POST, request.FILES)
        if not form1.is_valid():
            errors.append("Error: Form not valid.")
        else:
            data_file = request.FILES['file']
            print(os.getcwd())
            with open('collection/static/webservice files/db_backup/' + data_file.name, 'wb+') as destination:
                for chunk in data_file.chunks():
                    destination.write(chunk)

            request.FILES['file']
            errors.append("The file was successfully uploaded.")

    return render(request, 'admin/upload_file.html', {'errors': errors})



# ==================
# TEST PAGES
# ==================
def test_page1(request):
    # s3 = boto3.resource('s3')
    # obj = s3.Object('elasticbeanstalk-eu-central-1-662304246363', 'SimulationModels/simulation_' + str(self.simulation_id) + '_validation_data.json')
    # validation_data = json.loads(obj.get()['Body'].read().decode('utf-8'))


    simulation_models = Simulation_model.objects.all().order_by('id') 
    for simulation_model in simulation_models:
        learn_parameters_results = Learn_parameters_result.objects.filter(simulation_id=simulation_model.id)
        for learn_parameters_result in learn_parameters_results:
            learn_parameters_result.run_number = simulation_model.run_number
            learn_parameters_result.save()



    return HttpResponse('success')





    # return HttpResponse('success: ' + str(exists_query))




def test_page2(request):
    new_run_numbers_dict = {8:54, 9:55, 2:9, 10:10}
    for result_id in new_run_numbers_dict.keys():
        learn_parameters_result = Learn_parameters_result.objects.get(id=result_id)
        learn_parameters_result.run_number = new_run_numbers_dict[result_id]
        learn_parameters_result.save()

    return HttpResponse('success')





def test_page3(request):
    response = {}
    simulation_models = Simulation_model.objects.all()
    for simulation_model in simulation_models:
        objects_dict = json.loads(simulation_model.objects_dict)
        learned_rules = {}
        for object_number in objects_dict.keys():
            learned_rules[object_number] = {}
            for attribute_id in objects_dict[object_number]['object_rules'].keys():
                learned_rules[object_number][attribute_id] = {}
                for rule_id in objects_dict[object_number]['object_rules'][attribute_id].keys():
                    if objects_dict[object_number]['object_rules'][attribute_id][rule_id]['learn_posterior']:
                        learned_rules[object_number][attribute_id][rule_id] = True
                    else:
                        learned_rules[object_number][attribute_id][rule_id] = False
        response[str(simulation_model.id) + '-' + str(simulation_model.execution_order_id)] = json.dumps(learned_rules)



    # return HttpResponse(json.dumps(bla, sort_keys=True, cls=generally_useful_functions.SortedListEncoder))
    return HttpResponse(json.dumps(response))

    # return render(request, 'tool/test_page3.html')



    



