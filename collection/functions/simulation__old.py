####################################################################
# This file is part of the Tree of Knowledge project.
#
# Copyright (c) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, November 2024
#####################################################################

from collection.models import Simulation_model, Rule, Likelihood_fuction, Attribute, Execution_order, Rule_parameter
import json
import pandas as pd
import numpy as np
from collection.functions import query_datapoints, get_from_db, generally_useful_functions
from operator import itemgetter
import random
import random
from scipy.stats import rv_histogram
import math
from copy import deepcopy
import re
import traceback
import pdb
import boto3
import psycopg2
import time

# called from edit_model.html
class Simulator:
    """This class gets initialized with values specified in edit_simulation.html.
    This includes the initial values for some objects. 
    By running the simulation the values for the next timesteps are determined and 
    if possible compared to the values in the KB."""


    objects_dict = {}
    simulation_start_time = 946684800
    simulation_end_time = 1577836800
    timestep_size = 31622400

    times = []
    y0_columns = []
    y0_column_dt = {}
    parameter_columns = []
    rules = []
    currently_running_learn_likelihoods = False
    number_of_batches = 0





# =================================================================================================================
#   _____       _ _   _       _ _         
#  |_   _|     (_) | (_)     | (_)        
#    | |  _ __  _| |_ _  __ _| |_ _______ 
#    | | | '_ \| | __| |/ _` | | |_  / _ \
#   _| |_| | | | | |_| | (_| | | |/ /  __/
#  |_____|_| |_|_|\__|_|\__,_|_|_/___\___|
# 
# =================================================================================================================

    def __init__(self, simulation_id):

        # IMPORTANT SETTINGS  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        
        limit_to_populated_y0_columns = True
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        



        self.simulation_id = simulation_id

        simulation_model_record = Simulation_model.objects.get(id=simulation_id)

        self.run_number = simulation_model_record.run_number + 1
        self.objects_dict = json.loads(simulation_model_record.objects_dict)
        self.execution_order_id = simulation_model_record.execution_order_id
        self.environment_start_time = simulation_model_record.environment_start_time
        self.environment_end_time = simulation_model_record.environment_end_time
        self.simulation_start_time = simulation_model_record.simulation_start_time
        self.simulation_end_time = simulation_model_record.simulation_end_time
        self.timestep_size = simulation_model_record.timestep_size  
        self.is_timeseries_analysis = simulation_model_record.is_timeseries_analysis
        self.nb_of_tested_parameters = simulation_model_record.nb_of_tested_parameters
        self.nb_of_parameters_to_keep = simulation_model_record.nb_of_parameters_to_keep
        self.max_number_of_instances = simulation_model_record.max_number_of_instances
        self.error_threshold = simulation_model_record.error_threshold
        self.run_locally = simulation_model_record.run_locally
        execution_order = json.loads(Execution_order.objects.get(id=self.execution_order_id).execution_order)

        if not self.is_timeseries_analysis:
            self.timestep_size = self.simulation_end_time - self.simulation_start_time




        # logging
        self.progress_tracking_file_name = 'collection/static/webservice files/runtime_data/simulation_progress_' + str(self.simulation_id) + '.txt'
        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 1, "total_number": 6}))





        #  ================  GET DATA  ===========================================



        #  --- y0_columns & y0_column_dt ---
        y_value_attributes = json.loads(simulation_model_record.y_value_attributes)
        for y_value_attribute in generally_useful_functions.deduplicate_list_of_dicts(y_value_attributes):
            column_name = 'obj' + str(y_value_attribute['object_number']) + 'attr' + str(y_value_attribute['attribute_id'])
            self.y0_columns.append(column_name)
            self.y0_column_dt[column_name] = Attribute.objects.get(id=y_value_attribute['attribute_id']).data_type
        self.y0_columns = list(set(self.y0_columns))


        #  --- times ---
        if self.is_timeseries_analysis:
            self.times = generally_useful_functions.get_list_of_times(self.simulation_start_time, self.simulation_end_time, self.timestep_size)
        else:
            self.times = [self.simulation_start_time, self.simulation_start_time]



        #  --- df & y0_values ---
        validation_data = json.loads(simulation_model_record.validation_data)  
        reduced_objects_dict = {}
        for object_number in self.objects_dict.keys():
            reduced_objects_dict[object_number] = {'object_filter_facts':self.objects_dict[object_number]['object_filter_facts'], 'object_relations':self.objects_dict[object_number]['object_relations'] }
        new_simulation_state_code = str(self.is_timeseries_analysis) + str(reduced_objects_dict) + str(self.simulation_start_time) + str(self.simulation_end_time) + str(self.timestep_size) + str(self.y0_columns) + str(self.max_number_of_instances)
        if 'simulation_state_code' in validation_data.keys():
            print('NEW QUERY?  %s == %s'  % (validation_data['simulation_state_code'], new_simulation_state_code))
        if 'simulation_state_code' in validation_data.keys() and validation_data['simulation_state_code'] == new_simulation_state_code:
            self.df = pd.DataFrame.from_dict(validation_data['df'])
            self.y0_values = validation_data['y0_values']
        else:
            (self.df, self.y0_values) = self.get_new_df_and_y0_values(self.is_timeseries_analysis, self.objects_dict, self.simulation_start_time, self.simulation_end_time, self.timestep_size, limit_to_populated_y0_columns, self.times, self.y0_columns, self.max_number_of_instances)
            validation_data = {'simulation_state_code': new_simulation_state_code,
                                'df': self.df.to_dict(orient='list'),
                                'y0_values':self.y0_values}
            simulation_model_record.validation_data = json.dumps(validation_data)
        
        simulation_model_record.run_number = self.run_number
        simulation_model_record.aborted = False

        simulation_model_record.save()
        self.y0_values_df = pd.DataFrame(self.y0_values)
        self.easy_to_fulfill_simulations = np.zeros(len(self.df))








        #  ================  PREPARE RULES  ===========================================

        # preparation: put relations into a dictionary for easier access
        relation_dict = {}
        object_numbers = self.objects_dict.keys()
        for object_number in object_numbers:
            relation_dict[object_number] = {}
            for relation in self.objects_dict[str(object_number)]['object_relations']:
                if relation['attribute_id'] not in relation_dict[object_number]:
                    relation_dict[object_number][relation['attribute_id']] = [relation['target_object_number']]
                else:
                    relation_dict[object_number][relation['attribute_id']] += [relation['target_object_number']]



        
        self.rules = []
        self.not_used_rules = {object_number:{} for object_number in object_numbers}

        for object_number in object_numbers:

            # object_attribute_ids = self.objects_dict[str(object_number)]['object_rules'].keys()
            # object_type_id = self.objects_dict[str(object_number)]['object_type_id']
            # attribute_ids = [attr['id'] for attr in execution_order['attribute_execution_order'][object_type_id]['used_attributes'] if attr['id'] in object_attribute_ids]
            object_type_id = self.objects_dict[str(object_number)]['object_type_id']
            attribute_ids = [attr['id'] for attr in execution_order['attribute_execution_order'][object_type_id]['used_attributes']]

            for attribute_id in attribute_ids:

                rule_ids = execution_order['rule_execution_order'][str(attribute_id)]['used_rule_ids']
                for rule_id in set(rule_ids):
                    print(str(object_number) + ', ' + str(attribute_id) + ', ' + str(rule_id))

                    try:
                        rule = self.objects_dict[str(object_number)]['object_rules'][str(attribute_id)][str(rule_id)]

                        # STEP 1: adapt condition_exec and effect_exec to current df  ----------------

                        # if set(rule['used_attribute_ids']) <= set(selcondition_execf.df.columns): # the attributes used in this rule must appear in df
                        if self.is_timeseries_analysis or 'df.delta_t' not in rule['effect_exec']:  # don't include rules containing delta_t for cross-sectional analyses 
                            if rule['effect_is_calculation']:
                                rule['effect_exec'] = rule['effect_exec'].replace('df.attr', 'df.obj' + str(object_number) + 'attr')
                            elif rule['changed_var_data_type'] in ['relation','int']:
                                rule['effect_exec'] = int(rule['effect_exec'])
                            elif rule['changed_var_data_type'] == 'real':
                                rule['effect_exec'] = float(rule['effect_exec'])
                            elif rule['changed_var_data_type'] in ['boolean','bool']:
                                rule['effect_exec'] = (rule['effect_exec'] in ['True','true','T','t'])
                            

                            # --- convert condition_exec ---
                            if not rule['is_conditionless']:

                                # Relations - condition --------------------------------------
                                # first level
                                relation_occurences = re.findall(r'df.rel\d+\.', rule['condition_exec'])
                                for relation_occurence in relation_occurences:
                                    relation_id = int(re.findall(r'\d+', relation_occurence)[0]) 
                                    if relation_id in relation_dict[object_number].keys():
                                        target_object_number = relation_dict[object_number][relation_id][0]
                                        rule['condition_exec'] = rule['condition_exec'].replace(relation_occurence, 'df.obj' + str(target_object_number))
                                    else: 
                                        relation_name = Attribute.objects.get(id=relation_id).name
                                        raise Exception(self.objects_dict[object_number]['object_name'] +  " doesn't have the relation '" + relation_name + "'")


                                # further levels
                                for level in range(2): # you can maximally have a relation of a relation of a relation (=3)
                                    relation_occurences = re.findall(r'df.obj\d+rel\d+\.', rule['condition_exec'])
                                    for relation_occurence in relation_occurences:
                                        given_object_number = int(re.findall(r'\d+', relation_occurence)[0]) 
                                        relation_id = int(re.findall(r'\d+', relation_occurence)[1]) 
                                        target_object_number = relation_dict[given_object_number][relation_id][0]
                                        rule['condition_exec'] = rule['condition_exec'].replace(relation_occurence, 'df.obj' + str(target_object_number))


                            rule['condition_exec'] = rule['condition_exec'].replace('df.attr', 'df.obj' + str(object_number) + 'attr')
                            rule['column_to_change'] = 'obj' + str(object_number) + 'attr' + str(rule['changed_var_attribute_id'])
                            rule['object_number'] = object_number

                            rule['parameters'] = {}
                            for used_parameter_id in rule['used_parameter_ids']:
                                parameter = Rule_parameter.objects.get(id=used_parameter_id)
                                rule['parameters'][used_parameter_id] = {'min_value': parameter.min_value, 'max_value': parameter.max_value}


                            # ---  parameter_columns  ---
                            if rule['learn_posterior']:
                                # rule probability
                                if not rule['has_probability_1']:
                                    self.parameter_columns.append('triggerThresholdForRule' + str(rule_id))
                                # parameter
                                for used_parameter_id in rule['used_parameter_ids']:
                                    self.parameter_columns.append('param' + str(used_parameter_id))




                            # ----  aggregation  ---
                            #if len(rule['aggregation_exec']) > 0:
                            #    used_objects = []
                            #    for agg_object_number in object_numbers:
                            #        required_object_columns = ['obj' + str(agg_object_number) + 'attr' + str(attribute_id) for attribute_id in rule['used_attribute_ids']]
                            #        if (set(required_object_columns) <= set(self.df.columns  + ['df.delta_t', 'df.randomNumber'])):
                            #            used_objects.append(agg_object_number)

                            #    if len(used_objects) > 0:
                            #        rule['aggregation_exec'].replace('x.', 'df.obj' + used_object)
                            #        ['(%s)'  used_object in used_objects]
                            #    else:
                            #        raise Exception("None of the objects have all the columns required by this rule, which are: "  + str(rule['used_attribute_ids']))

                            #exists_expressions = re.findall(r'\([∀∃]rel\d+\)\[[^\]]*\]', rule['condition_exec'])
                            #for exists_expression in exists_expressions:
                            #    relation_id = int(re.findall(r'\d+', exists_expression)[0])
                            #    target_object_numbers = relation_dict[object_number][relation_id]
                            #    exists_expression_inner = re.findall(r'\[.*\]',exists_expression)[0]
                            #    list_of_different_inner_expressions = [exists_expression_inner.replace('df.rel' + str(relation_id), 'df.obj'+ str(target_object_number)) for target_object_number in target_object_numbers]
                            #    if exists_expression[1]=='∃':
                            #        replacement = '(' + ' or '.join(list_of_different_inner_expressions) + ')'
                            #    else:
                            #        replacement = '(' + ' and '.join(list_of_different_inner_expressions) + ')'
                            #    rule['condition_exec'] = rule['condition_exec'].replace(exists_expression, replacement)






                            # ---  histograms  ---
                            # rule probability
                            if (not rule['has_probability_1']) and (not rule['learn_posterior']):
                                # if a specific posterior for this simulation has been learned, take this, else take the combined posterior of all other simulations
                                histogram, mean, standard_dev, nb_of_values_in_posterior, message= get_from_db.get_single_pdf(simulation_id, object_number, rule_id, True)
                                if histogram is None:
                                    histogram, mean, standard_dev, nb_of_values_in_posterior, nb_of_simulations = get_from_db.get_rules_pdf(rule_id, True)
                                rule['histogram'] = histogram

                            # parameter
                            if not rule['learn_posterior']:
                                for used_parameter_id in rule['used_parameter_ids']: 
                                    histogram, mean, standard_dev, nb_of_values_in_posterior, message= get_from_db.get_single_pdf(simulation_id, object_number, used_parameter_id, False)
                                    if histogram is None:
                                        histogram, mean, standard_dev, nb_of_values_in_posterior, nb_of_simulations = get_from_db.get_rules_pdf(used_parameter_id, False)

                                    # change to the parameter's range
                                    min_value = rule['parameters'][used_parameter_id]['min_value']
                                    max_value = rule['parameters'][used_parameter_id]['max_value']
                                    histogram = (histogram[0], np.linspace(min_value,max_value,31))
                                    rule['parameters'][used_parameter_id]['histogram'] = histogram
                                    self.posterior_values['param' + str(used_parameter_id)] = []


                            # used_columns
                            used_columns = re.findall(r'df\.[^ \(\)\*\+\-\.\"\']*', rule['condition_exec'])
                            used_columns = [col.replace('df.','') for col in used_columns]
                            rule['used_columns'] = used_columns
                            rule['condition_exec'] = rule['condition_exec'].replace('df.', 'populated_df.')

                            # check if all the mentioned columns appear in df
                            mentioned_columns = re.findall(r'df\.[a-zA-Z0-9_]+', rule['condition_exec'] + ' ' + str(rule['effect_exec']))
                            mentioned_columns = [col for col in mentioned_columns if col[:8] != 'df.param']
                            mentioned_columns += ['df.' + rule['column_to_change']]
                            df_columns = ['df.'+col for col in self.df.columns]
                            if (set(mentioned_columns) <= set(df_columns + ['df.delta_t', 'df.randomNumber'])):
                                self.rules.append(rule)
                            else: 
                                raise Exception("The following columns are missing: " + str(list(set(mentioned_columns) - set(df_columns + ['df.delta_t']))))

                    except Exception:
                        self.not_used_rules[object_number][rule_id] = {'condition_text':rule['condition_text'], 
                                                                        'effect_text':rule['effect_text'], 
                                                                        'reason':str(traceback.format_exc())}

        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 6, "total_number": 6}))







    def get_new_df_and_y0_values(self, is_timeseries_analysis, objects_dict, simulation_start_time, simulation_end_time, timestep_size, limit_to_populated_y0_columns, times, y0_columns, max_number_of_instances):

        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 2, "total_number": 6}))

        if limit_to_populated_y0_columns:
            all_periods_df = query_datapoints.get_data_from_related_objects__multiple_timesteps(objects_dict, simulation_start_time, simulation_end_time, timestep_size, self.progress_tracking_file_name, y0_columns=y0_columns)
        else: 
            all_periods_df = query_datapoints.get_data_from_related_objects__multiple_timesteps(objects_dict, simulation_start_time, simulation_end_time, timestep_size, self.progress_tracking_file_name)

        y0_values = []
        if is_timeseries_analysis:
            
            all_periods_df = self.reduce_number_of_rows(all_periods_df, max_number_of_instances)
            all_periods_df.index = range(len(all_periods_df))

            # coalesce the first periods into the starting values (i.e. self.df)
            number_of_periods_in_df = int(np.ceil(len(times)/3)) 
            object_id_columns = [col for col in all_periods_df.columns if 'object_id' in col]
            df = all_periods_df[object_id_columns]
            attribute_columns = set([col.split('period')[0] for col in all_periods_df.columns if 'period' in col])
            for attribute_column in attribute_columns:
                the_attributes_periods__tuples = [(col, int(col.split('period')[1])) for col in all_periods_df.columns if col.split('period')[0]==attribute_column and int(col.split('period')[1])<=number_of_periods_in_df]
                the_attributes_periods = [period[0] for period in sorted(the_attributes_periods__tuples, key=lambda tup: tup[1])] 
                if len(the_attributes_periods) > 0:
                    df[attribute_column] = all_periods_df[the_attributes_periods].values[np.arange(len(all_periods_df)), np.argmin(pd.isnull(all_periods_df[the_attributes_periods]).values, axis=1)]

            for col in y0_columns:
                desired_column_names = [col + 'period'+ str(period) for period in range(len(times))]
                for desired_column_name in desired_column_names:
                    if desired_column_name not in all_periods_df.columns:
                        all_periods_df[desired_column_name] = np.nan

            
            all_periods_df = pd.merge(all_periods_df, df, on=object_id_columns)
            all_periods_df = all_periods_df[[col for col in all_periods_df.columns if col.split('period')[0] in y0_columns]]
            y0_values = [row for index, row in sorted(all_periods_df.to_dict('index').items())]


        else:
            df = all_periods_df
            df.columns = [col.split('period')[0] for col in df.columns]
            df = self.reduce_number_of_rows(df, max_number_of_instances)
            df_copy = pd.DataFrame(df[y0_columns].copy())
            df_copy.columns = [col + 'period0' for col in df_copy.columns]
            df_copy = df_copy[[col for col in df_copy.columns if col.split('period')[0] in y0_columns]]
            y0_values = [row for index, row in sorted(df_copy.to_dict('index').items())]


        df.fillna(value=pd.np.nan, inplace=True)
        df.index = range(len(df))

        return (df, y0_values)

# ==========================================================================================
#    __  __       _       
#   |  \/  |     (_)      
#   | \  / | __ _ _ _ __  
#   | |\/| |/ _` | | '_ \ 
#   | |  | | (_| | | | | |
#   |_|  |_|\__,_|_|_| |_|
# 
# ==========================================================================================

    def run(self):
        



        self.__learn_likelihoods()


        (simulation_data_df, triggered_rules_df, errors_df) = self.__run_monte_carlo_simulation(300)
        self.__post_process_data(simulation_data_df, triggered_rules_df, errors_df, 300)







    def __learn_likelihoods(self):
        print('=======  learn_likelihoods  =======')
        # PART 1 - Run the Simulation
        self.currently_running_learn_likelihoods = True
        batch_size = len(self.df)

        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_dict_string = json.dumps({"learning_likelihoods": True, "nb_of_accepted_simulations_total": self.nb_of_tested_parameters * len(self.df), "nb_of_accepted_simulations_current": 0,  "learning__post_processing": "" , "running_monte_carlo": False })
            progress_tracking_file.write(progress_dict_string)


        all_priors_df = pd.DataFrame()
        self.nb_of_sim_in_which_rule_was_used = 0
        for rule in self.rules:
            if rule['learn_posterior']:
                all_priors_df['nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] = [np.nan] * self.nb_of_tested_parameters
                all_priors_df['error_rule' + str(rule['id'])] = [np.nan] * self.nb_of_tested_parameters
                if not rule['has_probability_1']:
                    all_priors_df['triggerThresholdForRule' + str(rule['id'])] = np.random.uniform(0, 1, self.nb_of_tested_parameters)
                for used_parameter_id in rule['used_parameter_ids']:
                    all_priors_df['param' + str(used_parameter_id)] = np.random.uniform(rule['parameters'][used_parameter_id]['min_value'], rule['parameters'][used_parameter_id]['max_value'], self.nb_of_tested_parameters)
        



        if len(all_priors_df) > 0:
            if self.run_locally:
                # =================  Simulation Loop  ==========================
                for batch_number in range(self.nb_of_tested_parameters):
                    print('learn_likelihoods (%s/%s)' % (batch_number+1, self.nb_of_tested_parameters))

                    with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
                        progress_tracking_file.write(json.dumps({"text": 'Learning parameters - simulation: ', "current_number": batch_number * len(self.df), "total_number": self.nb_of_tested_parameters * len(self.df)}))

                    priors_dict = all_priors_df.loc[batch_number,:].to_dict()

                    y0_values_in_simulation = self.likelihood_learning_simulator(self.df, self.rules, priors_dict, batch_size)
                    errors_dict = self.n_dimensional_distance(y0_values_in_simulation, self.y0_values) 
                    for rule in self.rules:
                        if rule['learn_posterior']:
                            all_priors_df.loc[batch_number, 'nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] = errors_dict[rule['id']]['nb_of_sim_in_which_rule_was_used']
                            all_priors_df.loc[batch_number,'error_rule' + str(rule['id'])] = errors_dict[rule['id']]['error'] 
                # ==============================================================
            else:
                for batch_number in range(self.nb_of_tested_parameters):
                    print('posting batch %s : %s' % (batch_number, str(all_priors_df.loc[batch_number,:].to_dict())))
                    simulation_parameters = {'simulation_id':  self.simulation_id, 'run_number':  self.run_number, 'batch_number': batch_number, 'rules': self.rules , 'priors_dict':  all_priors_df.loc[batch_number,:].to_dict(), 'batch_size': batch_size , 'is_timeseries_analysis': self.is_timeseries_analysis, 'times': self.times, 'timestep_size':  self.timestep_size, 'y0_columns': self.y0_columns, 'parameter_columns':  self.parameter_columns, 'y0_column_dt':  self.y0_column_dt, 'error_threshold':  self.error_threshold}
                    sqs = boto3.client('sqs', region_name='eu-central-1')
                    queue_url = 'https://sqs.eu-central-1.amazonaws.com/662304246363/Treeofknowledge-queue'
                    response = sqs.send_message(QueueUrl= queue_url, MessageBody=json.dumps(simulation_parameters))

                result_checking_start_time = time.time()
                connection = psycopg2.connect(user="dbadmin", password="rUWFidoMnk0SulVl4u9C", host="aa1pbfgh471h051.cee9izytbdnd.eu-central-1.rds.amazonaws.com", port="5432", database="ebdb")
                cursor = connection.cursor()
                all_simulation_results = []
                while (time.time() - result_checking_start_time < 36000):

                    time.sleep(1)
                    cursor.execute('''SELECT simulation_id, run_number, batch_number, priors_dict, simulation_results FROM tested_simulation_parameters WHERE simulation_id=%s AND run_number=%s;''' % (self.simulation_id, self.run_number))
                    all_simulation_results = cursor.fetchall() 
                    print('checking results - found %s/%s' % (len(all_simulation_results), self.nb_of_tested_parameters))

                    with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
                        progress_tracking_file.write(json.dumps({"text": 'Learning parameters - simulation: ', "current_number": len(all_simulation_results) * len(self.df), "total_number": self.nb_of_tested_parameters * len(self.df)}))

                    if Simulation_model.objects.get(id=self.simulation_id).aborted:
                        break;

                    if len(all_simulation_results) >= (self.nb_of_tested_parameters-1):
                        cursor.execute('''DELETE FROM tested_simulation_parameters WHERE simulation_id=%s AND run_number=%s;''' % (self.simulation_id, self.run_number))
                        break



                all_simulation_results_df = pd.DataFrame(all_simulation_results, columns=['simulation_id', 'run_number', 'batch_number', 'priors_dict', 'simulation_results'])
                for index, row in all_simulation_results_df.iterrows():
                    simulation_results = json.loads(row['simulation_results'])
                    for rule in self.rules:
                        if rule['learn_posterior']:
                            all_priors_df.loc[row['batch_number'], 'nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] = simulation_results['nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] 
                            all_priors_df.loc[row['batch_number'],'error_rule' + str(rule['id'])] = simulation_results['error_rule' + str(rule['id'])]


            
            # PART 2 - Post Processing
            self.currently_running_learn_likelihoods = False
            with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
                progress_tracking_file.write(json.dumps({"text": 'Learning parameters - post-processing: ', "current_number": self.nb_of_tested_parameters * len(self.df), "total_number": self.nb_of_tested_parameters * len(self.df)}))

            for rule_number, rule in enumerate(self.rules):

                # histogram
                print('rule%s.learn_posterior = %s; rule%s.has_probability_1 = %s;' % (rule['id'], rule['learn_posterior'], rule['id'], rule['has_probability_1']))
                if rule['learn_posterior']:
                    all_priors_df = all_priors_df.sort_values('error_rule' + str(rule['id']))
                    all_priors_df.index = range(len(all_priors_df))
                    priors_to_keep_df = all_priors_df[:self.nb_of_parameters_to_keep]
                    print('nb_of_parameters_to_keep: %s ; len(priors_to_keep_df): %s' % (self.nb_of_parameters_to_keep, len(priors_to_keep_df)))

                    if not rule['has_probability_1']:
                        print('==== post-processing rule'+ str(rule['id']) + '  ===================================================')
 
                        histogram  = np.histogram(list(priors_to_keep_df['triggerThresholdForRule'+ str(rule['id'])]), bins=30, range=(0.0,1.0))
                        print('histogram: ' + str(histogram))

                        # nb_of_simulations, nb_of_sim_in_which_rule_was_used, nb_of_values_in_posterior
                        nb_of_simulations = self.nb_of_tested_parameters * len(self.df)
                        nb_of_sim_in_which_rule_was_used = priors_to_keep_df['nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'].sum()
                        nb_of_values_in_posterior = len(priors_to_keep_df)

                        # PART 2.1: update the rule's histogram - the next simulation will use the newly learned probabilities
                        self.rules[rule_number]['histogram'] = histogram 

                        # PART 2.2: save the learned likelihood function to the database
                        list_of_probabilities_str = json.dumps(list( np.array(histogram[0]) * 30/ np.sum(histogram[0])))
                        print('list_of_probabilities_str: ' + list_of_probabilities_str)

                        try:
                            likelihood_fuction = Likelihood_fuction.objects.get(simulation_id=self.simulation_id, rule_id=rule['id'])
                            likelihood_fuction.list_of_probabilities = list_of_probabilities_str
                            likelihood_fuction.nb_of_simulations = nb_of_simulations
                            likelihood_fuction.nb_of_sim_in_which_rule_was_used = nb_of_sim_in_which_rule_was_used
                            likelihood_fuction.nb_of_values_in_posterior = nb_of_values_in_posterior
                            likelihood_fuction.save()
                            print('saved to existing Likelihood_fuction ' + str(likelihood_fuction.id))

                        except:
                            likelihood_fuction = Likelihood_fuction(simulation_id=self.simulation_id, 
                                                                    object_number=rule['object_number'],
                                                                    rule_id=rule['id'], 
                                                                    list_of_probabilities=list_of_probabilities_str,
                                                                    nb_of_simulations=nb_of_simulations,
                                                                    nb_of_sim_in_which_rule_was_used=nb_of_sim_in_which_rule_was_used,
                                                                    nb_of_values_in_posterior=nb_of_values_in_posterior)
                            likelihood_fuction.save()
                            print('saved to new Likelihood_fuction ' + str(likelihood_fuction.id))


                    for used_parameter_id in rule['used_parameter_ids']:
                        print('==== post-processing parameter ' + str(used_parameter_id) + '  ===================================================')

                        min_value = rule['parameters'][used_parameter_id]['min_value']
                        max_value = rule['parameters'][used_parameter_id]['max_value']


                        histogram  = np.histogram(list(priors_to_keep_df['param' + str(used_parameter_id)]), bins=30, range=(min_value,max_value))
                        print('histogram: ' + str(histogram))

                        # nb_of_simulations, nb_of_sim_in_which_rule_was_used, nb_of_values_in_posterior
                        nb_of_simulations = self.nb_of_tested_parameters * len(self.df)
                        nb_of_sim_in_which_rule_was_used = priors_to_keep_df['nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'].sum()
                        nb_of_values_in_posterior = len(priors_to_keep_df)

                        # PART 2.1: update the rule's histogram - the next simulation will use the newly learned probabilities
                        self.rules[rule_number]['parameters'][used_parameter_id]['histogram'] = histogram 

                        # PART 2.2: save the learned likelihood function to the database
                        list_of_probabilities_str = json.dumps(list( np.array(histogram[0]) * 30/ np.sum(histogram[0])))
                        print('list_of_probabilities_str: ' + list_of_probabilities_str)

                        try:
                            likelihood_fuction = Likelihood_fuction.objects.get(simulation_id=self.simulation_id, parameter_id=used_parameter_id)
                            likelihood_fuction.list_of_probabilities = list_of_probabilities_str
                            likelihood_fuction.nb_of_simulations = nb_of_simulations
                            likelihood_fuction.nb_of_sim_in_which_rule_was_used = nb_of_sim_in_which_rule_was_used
                            likelihood_fuction.nb_of_values_in_posterior = nb_of_values_in_posterior
                            likelihood_fuction.save()
                            print('saved to existing Likelihood_fuction ' + str(likelihood_fuction.id))

                        except:
                            likelihood_fuction = Likelihood_fuction(simulation_id=self.simulation_id, 
                                                                    object_number=rule['object_number'],
                                                                    parameter_id=used_parameter_id,  
                                                                    list_of_probabilities=list_of_probabilities_str,
                                                                    nb_of_simulations=nb_of_simulations,
                                                                    nb_of_sim_in_which_rule_was_used=nb_of_sim_in_which_rule_was_used,
                                                                    nb_of_values_in_posterior=nb_of_values_in_posterior)
                            likelihood_fuction.save()
                            print('saved to new Likelihood_fuction ' + str(likelihood_fuction.id))



        





    def __post_process_data(self, simulation_data_df, triggered_rules_df, errors_df, number_of_simulations):

        print('process_data_1')
        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Preparing results - step: ', "current_number": 1, "total_number": 3}))


        # rule_infos
        print('process_data_2')
        triggered_rules_df = triggered_rules_df[triggered_rules_df['triggered_rule'].notnull()]
        rule_ids = [triggered_rule_info['id'] for triggered_rule_info  in list(triggered_rules_df['triggered_rule'])]
        rule_ids = list(set(rule_ids))
        rule_info_list = list(Rule.objects.filter(id__in=rule_ids).values())
        rule_infos = {}
        for rule in rule_info_list:
            rule_infos[rule['id']] = rule
        


        # triggered_rules
        print('process_data_3.0')
        triggered_rules_per_period = triggered_rules_df.groupby(['batch_number','initial_state_id','attribute_id','period']).aggregate({'initial_state_id':'first',
                                                                                                        'batch_number':'first',
                                                                                                        'attribute_id':'first',
                                                                                                        'period':'first',
                                                                                                        'triggered_rule':list,
                                                                                                        'correct_value':'first',})  
        attribute_dict = {attribute_id: {} for attribute_id in triggered_rules_df['attribute_id'].unique().tolist()}
        triggered_rules = {}
        print('process_data_3.1 - ' + str(len(triggered_rules_df['batch_number'].unique())) )
        for batch_number in triggered_rules_df['batch_number'].unique().tolist():
            for initial_state_id in triggered_rules_df['initial_state_id'].unique().tolist():
                triggered_rules[str(initial_state_id) + '-' + str(batch_number)] = deepcopy(attribute_dict)

        print('process_data_3.2 - ' + str(len(triggered_rules_per_period)))
        for index, row in triggered_rules_per_period.iterrows():
            triggered_rules[str(row['initial_state_id']) + '-' + str(row['batch_number'])][row['attribute_id']][int(row['period'])] = {'rules': row['triggered_rule'], 'correct_value': row['correct_value']}




        # simulation_data
        print('process_data_4')
        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Preparing results - step: ', "current_number": 2, "total_number": 3}))

        simulation_data = {}
        attribute_ids = [attr_id for attr_id in simulation_data_df.columns if attr_id not in ['batch_number','initial_state_id','attribute_id','period', 'randomNumber', 'cross_join_column']]
        aggregation_dict = {attr_id:list for attr_id in attribute_ids}
        aggregation_dict['batch_number'] = 'first'
        aggregation_dict['initial_state_id'] = 'first'
        simulation_data_per_entity_attribute = simulation_data_df.groupby(['batch_number','initial_state_id']).aggregate(aggregation_dict)
        simulation_data_per_entity_attribute['initial_state_id'] = simulation_data_per_entity_attribute['initial_state_id'].astype(int)
        simulation_data_per_entity_attribute['batch_number'] = simulation_data_per_entity_attribute['batch_number'].astype(int)

        print('process_data_4.1 - ' + str(len(simulation_data_per_entity_attribute)))
        for index, row in simulation_data_per_entity_attribute.iterrows():
            for attribute_id in attribute_ids:
                simulation_number = str(row['initial_state_id']) + '-' + str(row['batch_number'])
                if simulation_number not in simulation_data.keys():
                    simulation_data[str(row['initial_state_id']) + '-' + str(row['batch_number'])] = {}
                simulation_data[str(row['initial_state_id']) + '-' + str(row['batch_number'])][attribute_id] = row[attribute_id].copy()


        correct_values = self.y0_values_df.to_dict()


        # errors
        print('process_data_5')
        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Preparing results - step: ', "current_number": 3, "total_number": 3}))

        errors = {}
        errors['score'] = 1 - errors_df['error'].mean()
        errors['correct_runs'] = list(errors_df.loc[errors_df['error'] < self.error_threshold, 'simulation_number'])
        errors['false_runs'] = list(errors_df.loc[errors_df['error'] > self.error_threshold, 'simulation_number'])



        # Front-End too slow?
        print('process_data_6')
        number_of_megabytes =len(json.dumps(simulation_data))/1000000
        if number_of_megabytes > 3:
            number_of_simulations_to_keep = int(len(simulation_data) * 3 / number_of_megabytes)
            keys_to_keep = list(simulation_data.keys())[:number_of_simulations_to_keep]
            simulation_data = {key:value for key, value in simulation_data.items() if key in keys_to_keep}
            triggered_rules = {key:value for key, value in triggered_rules.items() if key in keys_to_keep}
            # simulation_data = {k: d[k]) for k in keys if k in d} simulation_data
            # triggered_rules = triggered_rules[:number_of_simulations_to_send]



        print('process_data_7')
        simulation_model_record = Simulation_model.objects.get(id=self.simulation_id)
        simulation_model_record.rule_infos = json.dumps(rule_infos)
        simulation_model_record.not_used_rules = self.not_used_rules
        simulation_model_record.triggered_rules = json.dumps(triggered_rules)
        simulation_model_record.simulation_data = json.dumps(simulation_data)
        simulation_model_record.correct_values = json.dumps(correct_values)
        simulation_model_record.errors = json.dumps(errors)
        simulation_model_record.save()







# ===========================================================================================================
 #   _____ _                 _       _   _               ______                _   _                 
 #  / ____(_)               | |     | | (_)             |  ____|              | | (_)                
 # | (___  _ _ __ ___  _   _| | __ _| |_ _  ___  _ __   | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
 #  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \  |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
 #  ____) | | | | | | | |_| | | (_| | |_| | (_) | | | | | |  | |_| | | | | (__| |_| | (_) | | | \__ \
 # |_____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/

# ===========================================================================================================



    #  Rule Learning  ---------------------------------------------------------------------------------
    def likelihood_learning_simulator(self, df_original, rules, priors_dict, batch_size):
        print('---- likelihood_learning_simulator ----')
        df = df_original.copy()


        for rule_nb in range(len(rules)):
            rules[rule_nb]['rule_was_used_in_simulation'] = [False]*batch_size
            rule = rules[rule_nb]

            if rule['learn_posterior']:
                if not rule['has_probability_1']:
                    df['triggerThresholdForRule' + str(rule['id'])] = priors_dict['triggerThresholdForRule' + str(rule['id'])]
                for used_parameter_id in rule['used_parameter_ids']:
                    df['param' + str(used_parameter_id)] = priors_dict['param' + str(used_parameter_id)]
            else:
                if not rule['has_probability_1']:
                    df['triggerThresholdForRule' + str(rule['id'])] =  rv_histogram(rule['histogram']).rvs(size=batch_size)
                for used_parameter_id in rule['used_parameter_ids']:
                    df['param' + str(used_parameter_id)] = rv_histogram(rule['parameters'][used_parameter_id]['histogram']).rvs(size=batch_size)


        if self.is_timeseries_analysis: 
            df['delta_t'] = self.timestep_size
        else:
            df[self.y0_columns] = None


        # if self.is_timeseries_analysis:    
            # times = np.arange(self.simulation_start_time + self.timestep_size, self.simulation_end_time, self.timestep_size)    
        #     times = generally_useful_functions.get_list_of_times(self.simulation_start_time + self.timestep_size, self.simulation_end_time, self.timestep_size)    
        # else:
        #     times = [self.simulation_start_time, self.simulation_end_time]


        y0_values_in_simulation = pd.DataFrame(index=range(batch_size))
        for period in range(len(self.times[1:])):
            df['randomNumber'] = np.random.random(batch_size)
            for rule in rules:
                populated_df_rows = pd.Series([True] * len(df))
                for used_column in rule['used_columns']:
                    populated_df_rows = populated_df_rows & ~df[used_column].isna()
                populated_df = df[populated_df_rows]

                if rule['is_conditionless']:
                    condition_satisfying_rows = pd.Series([True] * batch_size)
                    if rule['has_probability_1']:
                        satisfying_rows = pd.Series([True] * batch_size)
                    else:
                        satisfying_rows = pd.eval('df.randomNumber < df.triggerThresholdForRule' + str(rule['id']))
                        
                else:
                    condition_satisfying_rows = pd.Series([False] * batch_size)

                    if len(populated_df)==0:
                        satisfying_rows = populated_df_rows
                    if rule['has_probability_1']:
                        condition_satisfying_rows[populated_df_rows] = pd.eval(rule['condition_exec'])
                        # == Testing ========================================================================
                        if period == 30:
                            period_30_condition_df = pd.read_csv('C:/Users/l412/Documents/2 temporary stuff/2020-06-25/period_30_condition.csv')
                            # for col in ['param52', 'obj1attr185', 'param50', 'param51', 'param57', 'param58', 'param59', 'param60', 'obj1attr92', 'obj1attr90', 'randomNumber']:
                            for col in ['param61','param62',  'obj1attr185','obj1attr92', 'obj1attr90', 'randomNumber']:
                                period_30_condition_df['run' + str(self.number_of_batches)+ '_' + col] = df[col]
                                period_30_condition_df['run' + str(self.number_of_batches) + 'condition_satisfied'] = pd.eval(rule['condition_exec'])
                            period_30_condition_df.to_csv('C:/Users/l412/Documents/2 temporary stuff/2020-06-25/period_30_condition.csv', index=False)
                        # ==========================================================================
                        if condition_satisfying_rows.iloc[0] in [-1,-2]: #messy bug-fix for bug where eval returns -1 and -2 instead of True and False
                            condition_satisfying_rows += 2
                            condition_satisfying_rows = condition_satisfying_rows.astype(bool)
                        satisfying_rows = condition_satisfying_rows
                    else:
                        condition_satisfying_rows[populated_df_rows] = pd.eval(rule['condition_exec'])
                        triggered_rules = pd.eval('df.randomNumber < df.triggerThresholdForRule' + str(rule['id']))
                        satisfying_rows = condition_satisfying_rows & triggered_rules 

                    # fix for: conditions with randomNumber/param+ might be basically satisfied except for the random number
                    if 'df.randomNumber' in rule['condition_exec'] or 'df.param' in rule['condition_exec']:
                        condition_satisfying_rows = pd.Series([True] * batch_size)


                # --------  used rules  --------
                if rule['learn_posterior']:
                    rule['rule_was_used_in_simulation'] = rule['rule_was_used_in_simulation'] | condition_satisfying_rows


                # --------  THEN  --------
                if rule['effect_is_calculation']: 
                    new_values = pd.eval(rule['effect_exec'])
                    if rule['changed_var_data_type'] in ['relation','int']:
                        nan_rows = new_values.isnull()
                        new_values = new_values.fillna(0)
                        new_values = new_values.astype(int)
                        new_values[nan_rows] = np.nan
                    elif rule['changed_var_data_type'] == 'real':
                        new_values = new_values.astype(float)
                    # elif rule['changed_var_data_type'] in ['boolean','bool']:
                    elif rule['changed_var_data_type'] in ['string','date']:
                        nan_rows = new_values.isnull()
                        new_values = new_values.astype(str)
                        new_values[nan_rows] = np.nan

                else:
                    # new_values = rule['effect_exec']
                    new_values = pd.Series(json.loads(rule['effect_exec']) * batch_size)


                # df.loc[satisfying_rows,rule['column_to_change']] = new_values 
                satisfying_rows[satisfying_rows.isna()] = False
                new_values[np.logical_not(satisfying_rows)] = df.loc[np.logical_not(satisfying_rows),rule['column_to_change']]
                df[rule['column_to_change']] = new_values


            y0_values_in_this_period = pd.DataFrame(df[self.y0_columns])
            y0_values_in_this_period.columns = [col + 'period' + str(period+1) for col in y0_values_in_this_period.columns] #faster version
            y0_values_in_simulation = y0_values_in_simulation.join(y0_values_in_this_period)


        for rule in rules:  
            if rule['learn_posterior']:
                y0_values_in_simulation['rule_used_in_simulation_' + str(rule['id'])] = rule['rule_was_used_in_simulation']
                del rule['rule_was_used_in_simulation']


        y0_values_in_simulation = pd.concat([y0_values_in_simulation,df[self.parameter_columns]], axis=1)
        y0_values_in_simulation.index = range(len(y0_values_in_simulation))
        return y0_values_in_simulation.to_dict('records')

          





    #  Monte-Carlo  ---------------------------------------------------------------------------------
    def __run_monte_carlo_simulation(self, nb_of_simulations=300):
        print('¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬  __run_monte_carlo_simulation   ¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬')
        y0 = np.asarray(self.df[self.y0_columns].copy())
        batch_size = len(y0)


        triggered_rules_df = pd.DataFrame()
        errors_df = pd.DataFrame()

        number_of_batches = math.ceil(nb_of_simulations/batch_size)
        for batch_number in range(number_of_batches):


            with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
                progress_tracking_file.write(json.dumps({"text": 'Making predictions - simulation: ', "current_number": batch_number*batch_size, "total_number": nb_of_simulations}))


            df = self.df.copy()

            simulation_data_df = df.copy()
            simulation_data_df['initial_state_id'] = df.index
            simulation_data_df['batch_number'] = batch_number
            simulation_data_df['period'] = 0


            for rule in self.rules:
                if rule['changed_var_data_type'] in ['boolean','bool']:
                    simulation_data_df[rule['column_to_change']] = simulation_data_df[rule['column_to_change']].astype('object')
                if not rule['has_probability_1']:
                    df['triggerThresholdForRule' + str(rule['id'])] =  rv_histogram(rule['histogram']).rvs(size=batch_size)
                for used_parameter_id in rule['used_parameter_ids']:
                    df['param' + str(used_parameter_id)] = rv_histogram(rule['parameters'][used_parameter_id]['histogram']).rvs(size=batch_size)


            if self.is_timeseries_analysis: 
                df['delta_t'] = self.timestep_size
            else: 
                df[self.y0_columns] = None

            # if self.is_timeseries_analysis:
                # times = np.arange(self.simulation_start_time + self.timestep_size, self.simulation_end_time, self.timestep_size)
                # times = generally_useful_functions.get_list_of_times(self.simulation_start_time + self.timestep_size, self.simulation_end_time, self.timestep_size)
            # else:
                # times = [self.simulation_start_time, self.simulation_end_time]


            y0_values_in_simulation = pd.DataFrame(index=range(batch_size))
            for period in range(len(self.times[1:])):
                df['randomNumber'] = np.random.random(batch_size)
                for rule in self.rules:
                    populated_df_rows = pd.Series([True] * len(df))
                    for used_column in rule['used_columns']:
                        populated_df_rows = populated_df_rows & ~df[used_column].isna()
                    populated_df = df[populated_df_rows]

                    # Apply Rule  ================================================================
                    if rule['is_conditionless']:
                        if rule['has_probability_1']:
                            satisfying_rows = populated_df_rows
                            condition_satisfying_rows = [True] * batch_size
                            trigger_thresholds = [0] * batch_size
                        else:
                            satisfying_rows = populated_df_rows & pd.eval('df.randomNumber < df.triggerThresholdForRule' + str(rule['id'])).tolist()
                            condition_satisfying_rows = [True] * batch_size
                            trigger_thresholds = list(df['triggerThresholdForRule' + str(rule['id'])])

                    else:

                        condition_satisfying_rows = pd.Series([False] * batch_size)

                        if len(populated_df)==0:
                            satisfying_rows = populated_df_rows
                            trigger_thresholds = [0] * batch_size
                        elif rule['has_probability_1']:
                            condition_satisfying_rows[populated_df_rows] = pd.eval(rule['condition_exec'])
                            if condition_satisfying_rows.iloc[0] in [-1,-2]: #messy bug-fix for bug where eval returns -1 and -2 instead of True and False
                                condition_satisfying_rows += 2
                                condition_satisfying_rows = condition_satisfying_rows.astype(bool)
                            satisfying_rows = condition_satisfying_rows.tolist()
                            trigger_thresholds = [0] * batch_size
                        else:
                            condition_satisfying_rows[populated_df_rows] = pd.eval(rule['condition_exec'])
                            satisfying_rows = pd.eval('df.randomNumber < df.triggerThresholdForRule' + str(rule['id'])) & condition_satisfying_rows
                            trigger_thresholds = list(df['triggerThresholdForRule' + str(rule['id'])])
                


                    # --------  THEN  --------
                    if rule['effect_is_calculation']:
                        all_new_values = pd.eval(rule['effect_exec'])
                        if rule['changed_var_data_type'] in ['relation','int']:
                            nan_rows = all_new_values.isnull()
                            all_new_values = all_new_values.fillna(0)
                            all_new_values = all_new_values.astype(int)
                            all_new_values[nan_rows] = np.nan
                        elif rule['changed_var_data_type'] == 'real':
                            all_new_values = all_new_values.astype(float)
                        # elif rule['changed_var_data_type'] in ['boolean','bool']:
                        elif rule['changed_var_data_type'] in ['string','date']:
                            nan_rows = all_new_values.isnull()
                            all_new_values = all_new_values.astype(str)
                            all_new_values[nan_rows] = np.nan

                    else:
                        # all_new_values = [json.loads(rule['effect_exec'])] * batch_size
                        all_new_values = pd.Series([rule['effect_exec']] * batch_size)


                    # new_values = [value for value, satisfying in zip(all_new_values,satisfying_rows) if satisfying]
                    # df.loc[satisfying_rows,rule['column_to_change']] = new_values
                    new_values = all_new_values
                    new_values[np.logical_not(satisfying_rows)] = df.loc[np.logical_not(satisfying_rows),rule['column_to_change']]
                    df[rule['column_to_change']] = new_values


                    calculated_values = list(df[rule['column_to_change']])
                    errors = np.zeros(len(calculated_values))
                    correct_value = ['unknown'] * len(calculated_values)
                    if rule['column_to_change'] in self.y0_columns:
                        errors = self.error_of_single_values(np.array(calculated_values), rule['column_to_change'], period+1)
                        correct_value = list(self.y0_values_df[rule['column_to_change'] + 'period' + str(period+1)])


                    triggered_rule_infos_df = pd.DataFrame({'condition_satisfied': condition_satisfying_rows,
                                                            'id':[rule['id']]* batch_size,
                                                            'pt': satisfying_rows,          # pt = probability_triggered
                                                            'tp': trigger_thresholds,       # tp = trigger_probability
                                                            'v': calculated_values,         # v = new_value
                                                            'error':errors})

                    triggered_rule_infos = triggered_rule_infos_df.to_dict('records')
                    triggered_rule_infos = [rule_info if rule_info['condition_satisfied'] else None for rule_info in triggered_rule_infos]
                    for i in range(len(triggered_rule_infos)):
                        if triggered_rule_infos[i] is not None:
                                del triggered_rule_infos[i]['condition_satisfied']
                                if np.isnan(triggered_rule_infos[i]['error']):
                                    del triggered_rule_infos[i]['error']


                    currently_triggered_rules = pd.DataFrame({  'initial_state_id':df.index,
                                                                'batch_number':[batch_number]*batch_size,
                                                                'attribute_id':[rule['column_to_change']]*batch_size,
                                                                'period':[period+1]*batch_size,
                                                                'triggered_rule': triggered_rule_infos, 
                                                                'correct_value': correct_value
                                                                })

                    triggered_rules_df = triggered_rules_df.append(currently_triggered_rules)

                
                # simulated values
                df['initial_state_id'] = df.index
                df['batch_number'] = batch_number
                df['period'] = period+1
                simulation_data_df = simulation_data_df.append(df)

                # error
                y0_values_in_this_period = pd.DataFrame(df[self.y0_columns])
                y0_values_in_this_period.columns = [col + 'period' + str(period+1) for col in y0_values_in_this_period.columns] #faster version
                y0_values_in_simulation = y0_values_in_simulation.join(y0_values_in_this_period)


            errors_dict = self.n_dimensional_distance(y0_values_in_simulation.to_dict('records'), self.y0_values)
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print(str(errors_dict.keys()))
            print(len([str(index) + '-' + str(batch_nb) for index, batch_nb in zip(df.index, [batch_number]*len(df))]))
            print(len(errors_dict['all_errors']))
            error_df = pd.DataFrame({  'simulation_number': [str(index) + '-' + str(batch_nb) for index, batch_nb in zip(df.index, [batch_number]*len(df))],
                                        'error': errors_dict['all_errors']})
            errors_df = errors_df.append(error_df)



        return (simulation_data_df, triggered_rules_df, errors_df)











# ===========================================================================================================
#               _     _ _ _   _                   _    ______ _  __ _    _   _           _           
#      /\      | |   | (_) | (_)                 | |  |  ____| |/ _(_)  | \ | |         | |          
#     /  \   __| | __| |_| |_ _  ___  _ __   __ _| |  | |__  | | |_ _   |  \| | ___   __| | ___  ___ 
#    / /\ \ / _` |/ _` | | __| |/ _ \| '_ \ / _` | |  |  __| | |  _| |  | . ` |/ _ \ / _` |/ _ \/ __|
#   / ____ \ (_| | (_| | | |_| | (_) | | | | (_| | |  | |____| | | | |  | |\  | (_) | (_| |  __/\__ \
#  /_/    \_\__,_|\__,_|_|\__|_|\___/|_| |_|\__,_|_|  |______|_|_| |_|  |_| \_|\___/ \__,_|\___||___/
# 
# ===========================================================================================================

    def unchanged(self, y):
        return y



    def categorical_distance(self, u, v):
        u = np.asarray(u, dtype=object, order='c').squeeze()
        u = np.atleast_1d(u)
        v = np.asarray(v, dtype=object, order='c').squeeze()
        v = np.atleast_1d(v)
        u_v = 1. - np.equal(u, v).astype(int)
        return u_v





    def n_dimensional_distance(self, u, v):
        print('------------  n_dimensional_distance  ---------------------')
        # u = simulated values;  v = correct_values
        u = np.asarray(u, dtype=object, order='c').squeeze()
        u = np.atleast_1d(u)
        v = np.asarray(v, dtype=object, order='c').squeeze()
        v = np.atleast_1d(v)
        u_df = pd.DataFrame(list(u))
        v_df = pd.DataFrame(list(v))
        u_df = u_df.fillna(np.nan)
        v_df = v_df.fillna(np.nan)


        total_error = np.zeros(shape=len(u))
        dimensionality = np.zeros(shape=len(u))
        for y0_column in self.y0_columns:
            period_columns = [col for col in u_df.columns if col.split('period')[0] == y0_column]
            if self.y0_column_dt[y0_column] in ['string','bool','relation']:
                for period_column in period_columns:
                    error = 1. - np.equal(np.array(u_df[period_column]), np.array(v_df[period_column])).astype(int)
                    error[pd.isnull(v_df[period_column])] = 0 # set the error to zero where the correct value was not given
                    error[pd.isnull(u_df[period_column])] = 0 # set the error to zero where the simulation value was not given
                    total_error += error
                    dimensionality += 1 - np.array(np.logical_or(v_df[period_column].isnull(),u_df[period_column].isnull()).astype(int))
            elif self.y0_column_dt[y0_column] in ['int','real']:
                for period_column in period_columns:
                    period_number = max(int(period_column.split('period')[1]), 1)
      
                    # relative_change = np.abs(np.array(u_df[period_column]) - np.array(v_df[period_column.split('period')[0]]))/period_number
                    # normalisation_factor = np.maximum(np.abs(u_df[period_column]),np.abs(v_df[period_column.split('period')[0]]))
                    # normalisation_factor = np.maximum(normalisation_factor, 1)
                    # relative_change = relative_change/normalisation_factor
                    # # relative_change_non_null = np.nan_to_num(relative_change, nan=1.0)  

                    # absolute_change = np.abs(np.array(u_df[period_column]) - np.array(v_df[period_column.split('period')[0]]))
                    # absolute_change = absolute_change/np.abs(np.percentile(absolute_change, 30))
                    # # absolute_change_non_null = np.nan_to_num(absolute_change, nan=1.0) 

                    residuals = np.abs(np.array(u_df[period_column]) - np.array(v_df[period_column]))
                    non_null_residuals = residuals[~np.isnan(residuals)]
                    nth_percentile = np.percentile(non_null_residuals, self.error_threshold*100) if len(non_null_residuals) > 0 else 1# whereby n is the error_threshold. It therefore automatically adapts to the senistivity...
                    error_divisor = nth_percentile if nth_percentile != 0 else 1
                    error_in_error_range =  residuals/error_divisor
                    # pdb.set_trace()
                    error_in_error_range_non_null = np.nan_to_num(error_in_error_range, nan=0)  
                    error_in_error_range_non_null = np.minimum(error_in_error_range_non_null, 1)

                    true_change_factor = (np.array(v_df[period_column])/np.array(v_df[period_column.split('period')[0]]))
                    true_change_factor_per_period = np.power(true_change_factor, (1/period_number))
                    simulated_change_factor = (np.array(u_df[period_column])/np.array(v_df[period_column.split('period')[0]]))
                    simulated_change_factor_per_period = np.power(simulated_change_factor, (1/period_number))
                    error_of_value_change = np.abs(simulated_change_factor_per_period - true_change_factor_per_period) 
                    error_of_value_change_non_null = np.nan_to_num(error_of_value_change, nan=0)  
                    error_of_value_change_non_null = np.minimum(error_of_value_change_non_null, 1)

                    error = 0.5*np.minimum(error_in_error_range_non_null,error_of_value_change_non_null) + 0.25*np.sqrt(error_in_error_range_non_null) + 0.25*np.sqrt(error_of_value_change_non_null)
                    
                    null_value_places = np.logical_or(np.isnan(error_in_error_range), np.isnan(error_of_value_change))
                    error[null_value_places] = 0

                    dimensionality += 1 - null_value_places.astype('int')
                    total_error += error 

        non_validated_rows = dimensionality == 0
        dimensionality = np.maximum(dimensionality, [1]*len(u))
        error = total_error/dimensionality
        error[non_validated_rows] = 1


        errors_dict = {'all_errors':error}
        if self.currently_running_learn_likelihoods:
            for rule in self.rules:
                if rule['learn_posterior']:
                    rule_used_in_simulation = u_df['rule_used_in_simulation_' + str(rule['id'])]
                    errors_dict[rule['id']] = {'error': error[rule_used_in_simulation].mean(), 'nb_of_sim_in_which_rule_was_used': rule_used_in_simulation.sum()}

        return errors_dict



    def error_of_single_values(self, calculated_values, column_name, period):      
        initial_values = np.array(self.df[column_name])
        correct_values = np.array(self.y0_values_df[column_name + 'period' + str(period)])


        if self.y0_column_dt[column_name] in ['string','bool','relation']:
            errors = 1. - np.equal(np.array(calculated_values), np.array(correct_values)).astype(int)
        if self.y0_column_dt[column_name] in ['int','real']:
            residuals = np.abs(np.array(calculated_values) - np.array(correct_values))
            error_in_error_range =  1.5*residuals/np.max(residuals)
            error_in_error_range = np.nan_to_num(error_in_error_range, nan=1.0)  
            # error_in_value_range = residuals/(np.max(correct_values) - np.min(correct_values))

            true_change_percent_per_period = ((correct_values - initial_values)/initial_values)/max(period,1)
            simulated_change_percent_per_period = ((np.array(calculated_values) - initial_values)/initial_values)/max(period,1)
            error_of_value_change = np.minimum(np.abs(simulated_change_percent_per_period - true_change_percent_per_period) * 20,1)
            error_of_value_change = np.nan_to_num(error_of_value_change, nan=1.0)  

            errors = np.minimum(error_of_value_change, error_in_error_range)
            errors = np.minimum(errors, 1)

        return errors



    def reduce_number_of_rows(self, df, max_nb_of_rows):
        print('------------  reduce_number_of_rows  ---------------------------------')
        if len(df)> max_nb_of_rows:
            number_of_nulls_df = df.isnull().sum(1)
            actual_y0_columns = [col for col in df.columns if col.split('period')[0] in self.y0_columns]
            number_of_y0_nulls_df = df[actual_y0_columns].isnull().sum(1)
            score_df = 0.3 * (1 - number_of_nulls_df/max(number_of_nulls_df)) + 0.7 * (1 - number_of_y0_nulls_df/max(number_of_y0_nulls_df))

            reduced_df = score_df[score_df > 0.5]
            if len(reduced_df) > max_nb_of_rows: 
                # select random rows from amongst the rows with minimal score = 0.5
                df = df.loc[reduced_df.index]
                df = df.sample(n=max_nb_of_rows)
            # elif len(number_of_nulls_df[number_of_nulls_df < (max(number_of_nulls_df)/2)]) > max_nb_of_rows: 
            elif len(score_df[score_df > 0.25]) > max_nb_of_rows: 
                # select random rows from amongst the rows with minimal score = 0.25
                df = df.loc[score_df[score_df > 0.25].index]
                df = df.sample(n=max_nb_of_rows)
            else:
                # select random rows
                df = df.sample(n=max_nb_of_rows)

        return df




