####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (C) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, February 2021
#####################################################################

from collection.models import Simulation_model, Rule, Likelihood_function, Attribute, Execution_order, Rule_parameter, Monte_carlo_result, Learn_parameters_result
import json
import pandas as pd
import numpy as np
from collection.functions import query_datapoints, get_from_db, generally_useful_functions
from operator import itemgetter
import random
import random
from scipy.stats import rv_histogram, rankdata
import math
from copy import deepcopy
import re
import traceback
import pdb
import boto3
import psycopg2
import time
import boto3





# called from edit_model.html
class Simulator:
    """This class gets initialized with values specified in edit_simulation.html.
    This includes the initial values for some objects. 
    By running the simulation the values for the next timesteps are determined and 
    if possible compared to the values in the KB."""











# =================================================================================================================
#   _____       _ _   _       _ _         
#  |_   _|     (_) | (_)     | (_)        
#    | |  _ __  _| |_ _  __ _| |_ _______ 
#    | | | '_ \| | __| |/ _` | | |_  / _ \
#   _| |_| | | | | |_| | (_| | | |/ /  __/
#  |_____|_| |_|_|\__|_|\__,_|_|_/___\___|
# 
# =================================================================================================================

    def __init__(self, simulation_id, ignore_learn_posteriors):


        self.objects_dict = {}
        self.simulation_start_time = 946684800
        self.simulation_end_time = 1577836800
        self.timestep_size = 31622400

        self.times = []
        self.y0_columns = []
        self.y0_column_dt = {}
        self.parameter_columns = []
        self.rules = []
        self.currently_running_learn_likelihoods = False



        self.simulation_id = simulation_id
        simulation_model_record = Simulation_model.objects.get(id=simulation_id)

        if ignore_learn_posteriors:
            self.run_number = simulation_model_record.run_number
        else:
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
        self.max_number_of_instances = simulation_model_record.max_number_of_instances
        self.error_threshold = simulation_model_record.error_threshold
        self.run_locally = simulation_model_record.run_locally
        self.limit_to_populated_y0_columns = simulation_model_record.limit_to_populated_y0_columns
        execution_order = json.loads(Execution_order.objects.get(id=self.execution_order_id).execution_order)
        manually_set_initial_values = json.loads(simulation_model_record.manually_set_initial_values)

        if not self.is_timeseries_analysis:
            self.timestep_size = self.simulation_end_time - self.simulation_start_time




        # logging
        self.progress_tracking_file_name = 'collection/static/webservice files/runtime_data/simulation_progress_' + str(self.simulation_id) + '.txt'
        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 1, "total_number": 6}))





        #  ================  GET DATA  ===========================================

        #  --- y0_columns & y0_column_dt ---
        self.y_value_attributes = json.loads(simulation_model_record.y_value_attributes)
        for y_value_attribute in generally_useful_functions.deduplicate_list_of_dicts(self.y_value_attributes):
            column_name = 'obj' + str(y_value_attribute['object_number']) + 'attr' + str(y_value_attribute['attribute_id'])
            self.y0_columns.append(column_name)
            self.y0_column_dt[column_name] = Attribute.objects.get(id=y_value_attribute['attribute_id']).data_type
        self.y0_columns = sorted(set(self.y0_columns))


        #  --- times ---
        if self.is_timeseries_analysis:
            self.times = generally_useful_functions.get_list_of_times(self.simulation_start_time, self.simulation_end_time, self.timestep_size)
        else:
            self.times = [self.simulation_start_time, self.simulation_start_time]



        #  --- df & y0_values ---
        try:
            s3 = boto3.resource('s3')
            obj = s3.Object('elasticbeanstalk-eu-central-1-662304246363', 'SimulationModels/simulation_' + str(self.simulation_id) + '_validation_data.json')
            validation_data = json.loads(obj.get()['Body'].read().decode('utf-8'))
        except:
            validation_data = {'simulation_state_code': '','df':{}, 'y0_values':{}}
        reduced_objects_dict = {}
        for object_number in self.objects_dict.keys():
            reduced_objects_dict[object_number] = {'object_filter_facts':self.objects_dict[object_number]['object_filter_facts'], 'object_relations':self.objects_dict[object_number]['object_relations'] }
        new_simulation_state_code = str(self.is_timeseries_analysis) + '|' + str(self.simulation_start_time) + '|' + str(self.simulation_end_time) + '|' + str(self.timestep_size) + '|' + str(self.max_number_of_instances) + '|' + json.dumps(self.y0_columns, sort_keys=True, cls=generally_useful_functions.SortedListEncoder) + '|' + json.dumps(reduced_objects_dict, sort_keys=True, cls=generally_useful_functions.SortedListEncoder) + '|' + json.dumps(execution_order['attribute_execution_order'], sort_keys=True, cls=generally_useful_functions.SortedListEncoder) + '|' + json.dumps(manually_set_initial_values, sort_keys=True, cls=generally_useful_functions.SortedListEncoder) 
        if 'simulation_state_code' in validation_data.keys():
            print(str(validation_data['simulation_state_code'] == new_simulation_state_code))
            print('checking :')
            print(validation_data['simulation_state_code'])
            print('vs.')
            print(new_simulation_state_code)
        if 'simulation_state_code' in validation_data.keys() and validation_data['simulation_state_code'] == new_simulation_state_code:
            self.df = pd.DataFrame.from_dict(validation_data['df'])
            self.y0_values = validation_data['y0_values']
        else:
            (self.df, self.y0_values) = self.get_new_df_and_y0_values(self.is_timeseries_analysis, self.simulation_start_time, self.simulation_end_time, self.timestep_size, self.times, self.y0_columns, self.max_number_of_instances, self.objects_dict, execution_order['attribute_execution_order'], manually_set_initial_values)
            validation_data = {'simulation_state_code': new_simulation_state_code,
                                'df': self.df.to_dict(orient='list'),
                                'y0_values':self.y0_values}
            s3.Object('elasticbeanstalk-eu-central-1-662304246363', 'SimulationModels/simulation_' + str(self.simulation_id) + '_validation_data.json').put(Body=json.dumps(validation_data).encode('utf-8'))

        # ------------------------------  OLD  ----------------------------------------  
        # if 'simulation_state_code' in validation_data.keys() and validation_data['simulation_state_code'] == new_simulation_state_code:
        #     validation_data = json.loads(simulation_model_record.validation_data)  
        #     self.df = pd.DataFrame.from_dict(validation_data['df'])
        #     self.y0_values = validation_data['y0_values']
        # else:
        #     (self.df, self.y0_values) = self.get_new_df_and_y0_values(self.is_timeseries_analysis, self.simulation_start_time, self.simulation_end_time, self.timestep_size, self.times, self.y0_columns, self.max_number_of_instances, self.objects_dict, execution_order['attribute_execution_order'])
        #     validation_data = {'simulation_state_code': new_simulation_state_code,
        #                         'df': self.df.to_dict(orient='list'),
        #                         'y0_values':self.y0_values}
        #     simulation_model_record.validation_data = json.dumps(validation_data)
        # ----------------------------------------------------------------------------------
        
        simulation_model_record.run_number = self.run_number
        simulation_model_record.aborted = False

        simulation_model_record.save()
        self.y0_values_df = pd.DataFrame(self.y0_values)
        self.easy_to_fulfill_simulations = np.zeros(len(self.df))










        #  ================  PREPARE RULES  ===========================================
        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 6, "total_number": 6}))

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



        
        self.not_used_rules = {object_number:{} for object_number in object_numbers}

        for object_number in object_numbers:

            # object_attribute_ids = self.objects_dict[str(object_number)]['object_rules'].keys()
            # object_type_id = self.objects_dict[str(object_number)]['object_type_id']
            # attribute_ids = [attr['id'] for attr in execution_order['attribute_execution_order'][object_type_id]['used_attributes'] if attr['id'] in object_attribute_ids]
            object_type_id = self.objects_dict[str(object_number)]['object_type_id']
            attribute_ids = [attr['id'] for attr in execution_order['attribute_execution_order'][object_type_id]['used_attributes']]

            for attribute_id in attribute_ids:

                rule_ids = execution_order['rule_execution_order'][str(attribute_id)]['used_rule_ids']
                for rule_id in rule_ids:
                    print(str(object_number) + ', ' + str(attribute_id) + ', ' + str(rule_id))

                    try:
                        rule = self.objects_dict[str(object_number)]['object_rules'][str(attribute_id)][str(rule_id)]
                        print('rule %s:  learn_posterior=%s ;has_probability_1=%s; used_parameter_ids=%s' % (rule['id'], rule['learn_posterior'], rule['has_probability_1'], rule['used_parameter_ids']))


                        if not self.is_timeseries_analysis and 'df.delta_t' in rule['effect_exec']:  
                            raise Exception("Rules with delta_t only work for timeseries analyses.")


                        # ===  adapt rule to object  ===========
                        rule['object_number'] = object_number
                        rule['column_to_change'] = 'obj' + str(object_number) + 'attr' + str(rule['changed_var_attribute_id'])

                        # adapt 1: condition_exec
                        if not rule['is_conditionless']:
                            rule['condition_exec'] = self.collapse_relations(rule['condition_exec'], relation_dict, object_number)
                            rule['condition_exec'] = rule['condition_exec'].replace('df.attr', 'df.obj' + str(object_number) + 'attr')
                    


                        # adapt 2: aggregation_exec
                        if len(rule['aggregation_exec']) > 0:
                            rule['aggregation_exec'] = self.collapse_relations(rule['aggregation_exec'], relation_dict, object_number)
                            agg_cond_used_attributes = re.findall(r'x_df\.attr\d*', rule['aggregation_exec'])
                            agg_cond_used_attribute_ids = list(set([int(attr[9:]) for attr in agg_cond_used_attributes]))
                            used_objects = []
                            for agg_object_number in object_numbers:
                                required_object_columns = ['obj' + str(agg_object_number) + 'attr' + str(attribute_id) for attribute_id in agg_cond_used_attribute_ids]
                                if (set(required_object_columns) <= set(list(self.df.columns)  + ['df.delta_t', 'df.randomNumber'])):
                                    used_objects.append(agg_object_number)

                            if len(used_objects) > 0:
                                object_conditions = []
                                for used_object in used_objects:
                                    object_conditions.append('(%s)' % (rule['aggregation_exec'].replace('x_df.', 'df.obj' + used_object)))
                                count_x_occurences = re.findall(r'COUNT\(x\)', rule['effect_exec'])
                                for count_x_occurence in count_x_occurences:
                                    count_x_replacement_str = '(0 + %s)' % (' + 0 + '.join(object_conditions))
                                    rule['effect_exec'] = rule['effect_exec'].replace(count_x_occurence, count_x_replacement_str)
                                sum_occurences = re.findall(r'SUM\(.*\)', rule['effect_exec'])
                                if len(sum_occurences) > 0:
                                    rule['sums'] = {}
                                for sum_number, sum_occurence in enumerate(sum_occurences):
                                    sum_term = sum_occurence[3:]
                                    object_sum_terms = ['(0 + (' + object_condition + ')) * ' + sum_term.replace('x_df.', 'df.obj' + used_object) for used_object, object_condition in zip(used_objects,object_conditions)]
                                    object_sum_terms = [self.collapse_relations(sum_term, relation_dict, object_number) for sum_term in object_sum_terms]
                                    object_sum_terms = [sum_term.replace('df.attr', 'df.obj' + str(object_number) + 'attr') for sum_term in object_sum_terms]
                                    rule['sums'][sum_number] = object_sum_terms
                                    rule['effect_exec'] = rule['effect_exec'].replace(sum_occurence, ' (df.sum%s) ' % sum_number)
                            else:
                                raise Exception("None of the objects have all the columns required by this rule, which are: "  + str(rule['used_attribute_ids']))


                        # adapt 3: effect_exec
                        if rule['effect_is_calculation']:
                            rule['effect_exec'] = self.collapse_relations(rule['effect_exec'], relation_dict, object_number)
                            rule['effect_exec'] = rule['effect_exec'].replace('df.attr', 'df.obj' + str(object_number) + 'attr')
                        elif rule['changed_var_data_type'] in ['relation','int']:
                            rule['effect_exec'] = int(rule['effect_exec'])
                        elif rule['changed_var_data_type'] == 'real':
                            rule['effect_exec'] = float(rule['effect_exec'])
                        elif rule['changed_var_data_type'] in ['boolean','bool']:
                            rule['effect_exec'] = (rule['effect_exec'] in ['True','true','T','t'])




                        # ===  parameter_columns  ===========
                        rule['parameters'] = {}
                        for used_parameter_id in rule['used_parameter_ids']:
                            parameter = Rule_parameter.objects.get(id=used_parameter_id)
                            rule['parameters'][used_parameter_id] = {'min_value': parameter.min_value, 'max_value': parameter.max_value}

                        



                        # ===  histograms  ===========
                        # rule probability
                        if (not rule['has_probability_1']):
                            # if a specific posterior for this simulation has been learned, take this, else take the combined posterior of all other simulations
                            histogram, mean, standard_dev, nb_of_simulations, nb_of_sim_in_which_rule_was_used, nb_of_tested_parameters, nb_of_tested_parameters_in_posterior, histogram_smooth = get_from_db.get_rules_pdf(self.execution_order_id, rule_id, True)
                            if histogram_smooth is None:
                                if histogram is None:
                                    histogram, mean, standard_dev, nb_of_tested_parameters_in_posterior, message = get_from_db.get_single_pdf(simulation_id, self.execution_order_id, object_number, rule_id, True, True)
                                rule['histogram'] = (list(histogram[0]), list(histogram[1]))
                            else:    
                                rule['histogram'] = (list(histogram_smooth[0]), list(histogram_smooth[1]))

                        # parameter
                        if not rule['learn_posterior'] or ignore_learn_posteriors:
                            for used_parameter_id in rule['used_parameter_ids']: 
                                histogram, mean, standard_dev, nb_of_simulations, nb_of_sim_in_which_rule_was_used, nb_of_tested_parameters, nb_of_tested_parameters_in_posterior, histogram_smooth = get_from_db.get_rules_pdf(self.execution_order_id, used_parameter_id, False)
                                
                                histogram_to_use = None
                                if histogram_smooth is None:
                                    if histogram is None:
                                        histogram, mean, standard_dev, nb_of_sim_in_which_rule_was_used, message= get_from_db.get_single_pdf(simulation_id, self.execution_order_id, object_number, used_parameter_id, False, True)
                                        print('used_parameter_id:' + str(used_parameter_id) + ' - get_single_pdf:' + str(histogram))
                                    histogram_to_use = histogram
                                else:
                                    print('used_parameter_id:' + str(used_parameter_id) + ' - get_rules_pdf:' + str(histogram_to_use))
                                    histogram_to_use = histogram_smooth

                                # change to the parameter's range                                
                                min_value = rule['parameters'][used_parameter_id]['min_value']
                                max_value = rule['parameters'][used_parameter_id]['max_value']
                                rule['parameters'][used_parameter_id]['histogram'] = (list(histogram_to_use[0]), list(np.linspace(min_value,max_value,31)))


                        # used_columns
                        used_columns = re.findall(r'df\.[^ \(\)\*\+\-\.\"\']*', rule['condition_exec'])
                        used_columns = [col.replace('df.','') for col in used_columns]
                        rule['used_columns'] = used_columns
                        rule['condition_exec'] = rule['condition_exec'].replace('df.', 'populated_df.')

                        # check if all the mentioned columns appear in df
                        mentioned_columns = re.findall(r'df\.[a-zA-Z0-9_]+', rule['condition_exec'] + ' ' + str(rule['effect_exec']))
                        if 'sums' in rule:
                            for sum_number in rule['sums'].keys():
                                for sum_term in rule['sums'][sum_number]:
                                    mentioned_columns += re.findall(r'df\.[a-zA-Z0-9_]+', sum_term)
                        mentioned_columns = [col for col in mentioned_columns if col[:8] != 'df.param' and col[:6] != 'df.sum']
                        mentioned_columns += ['df.' + rule['column_to_change']]
                        df_columns = ['df.'+col for col in self.df.columns]
                        if (set(mentioned_columns) <= set(df_columns + ['df.delta_t', 'df.randomNumber'])):
                            self.rules.append(rule)
                        else: 
                            raise Exception("The following columns are missing: " + str(list(set(mentioned_columns) - set(df_columns + ['df.delta_t']))))

                    except Exception:
                        self.not_used_rules[object_number][rule_id] = {'condition_text':rule['condition_text'], 
                                                                        'condition_exec':rule['condition_exec'],
                                                                        'effect_text':rule['effect_text'], 
                                                                        'effect_exec':rule['effect_exec'], 
                                                                        'reason':str(traceback.format_exc())}
                                       
        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 6, "total_number": 6}))

        for rule in self.rules:
            if rule['learn_posterior']:
                if not rule['has_probability_1']:
                    self.parameter_columns.append('triggerThresholdForRule' + str(rule_id))
                for used_parameter_id in rule['used_parameter_ids']:
                    self.parameter_columns.append('param' + str(used_parameter_id))
        self.parameter_columns = list(set(self.parameter_columns))







    def collapse_relations(self, exec_text, relation_dict,object_number):
        # Relations - condition --------------------------------------
        # first level
        relation_occurences = re.findall(r'df.rel\d+\.', exec_text)
        for relation_occurence in relation_occurences:
            relation_id = int(re.findall(r'\d+', relation_occurence)[0]) 
            if relation_id in relation_dict[object_number].keys():
                target_object_number = relation_dict[object_number][relation_id][0]
                exec_text = exec_text.replace(relation_occurence, 'df.obj' + str(target_object_number))
            else: 
                relation_name = Attribute.objects.get(id=relation_id).name
                raise Exception(self.objects_dict[object_number]['object_name'] +  " doesn't have the relation '" + relation_name + "'")


        # further levels
        for level in range(2): # you can maximally have a relation of a relation of a relation (=3)
            relation_occurences = re.findall(r'df.obj\d+rel\d+\.', exec_text)
            for relation_occurence in relation_occurences:
                given_object_number = int(re.findall(r'\d+', relation_occurence)[0]) 
                relation_id = int(re.findall(r'\d+', relation_occurence)[1]) 
                if relation_id in relation_dict[given_object_number].keys():
                    target_object_number = relation_dict[given_object_number][relation_id][0]
                    exec_text = exec_text.replace(relation_occurence, 'df.obj' + str(target_object_number))
                else: 
                    relation_name = Attribute.objects.get(id=relation_id).name
                    raise Exception(self.objects_dict[given_object_number]['object_name'] +  " doesn't have the relation '" + relation_name + "'")

        return exec_text






    def get_new_df_and_y0_values(self, is_timeseries_analysis, simulation_start_time, simulation_end_time, timestep_size,  times, y0_columns, max_number_of_instances, objects_dict, attribute_execution_order, manually_set_initial_values):

        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 2, "total_number": 6}))

        if self.limit_to_populated_y0_columns:
            all_periods_df = query_datapoints.get_data_from_related_objects__multiple_timesteps(objects_dict, simulation_start_time, simulation_end_time, timestep_size, self.progress_tracking_file_name, max_number_of_instances, y0_columns=y0_columns)
        else: 
            all_periods_df = query_datapoints.get_data_from_related_objects__multiple_timesteps(objects_dict, simulation_start_time, simulation_end_time, timestep_size, self.progress_tracking_file_name, max_number_of_instances)

        y0_values_df = pd.DataFrame()
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
            y0_values_df = all_periods_df[[col for col in all_periods_df.columns if col.split('period')[0] in y0_columns]]


        else:
            df = all_periods_df
            df.columns = [col.split('period')[0] for col in df.columns]
            df = self.reduce_number_of_rows(df, max_number_of_instances)
            df_copy = pd.DataFrame(df[y0_columns].copy())
            df_copy.columns = [col + 'period0' for col in df_copy.columns]
            y0_values_df = df_copy[[col for col in df_copy.columns if col.split('period')[0] in y0_columns]]
            


        df.fillna(value=pd.np.nan, inplace=True)
        df.index = range(len(df))


        # add missing columns:
        all_wanted_columns = []
        all_not_wanted_columns = []
        for object_number in objects_dict.keys():
            object_type_id = objects_dict[object_number]['object_type_id']
            for wanted_attribute in attribute_execution_order[object_type_id]['used_attributes']:
                all_wanted_columns.append('obj' + str(object_number) + 'attr' + str(wanted_attribute['id']))
            for not_wanted_attribute in attribute_execution_order[object_type_id]['not_used_attributes']:
                all_not_wanted_columns.append('obj' + str(object_number) + 'attr' + str(not_wanted_attribute['id']))

        columns_to_add = list(set(all_wanted_columns) - set(df.columns))    
        for column_to_add in columns_to_add:
            df[column_to_add] = np.nan
            y0_values_df[column_to_add] = np.nan

        columns_to_remove = list(set(all_not_wanted_columns).intersection(set(df.columns)))
        for column_to_remove in columns_to_remove:
            del df[column_to_remove]
            if column_to_remove in y0_values_df.columns:
                del y0_values_df[column_to_remove] 

        # manually_set_initial_values
        for object_number in manually_set_initial_values.keys():
            for attribute_id in manually_set_initial_values[object_number].keys():
                df['obj' + str(object_number) + 'attr' + str(attribute_id)] = manually_set_initial_values[object_number][attribute_id]

        # some additional useful columns
        df['null'] = np.nan
        if is_timeseries_analysis: 
            df['delta_t'] = timestep_size
        else:
            df[y0_columns] = None

        y0_values = [row for index, row in sorted(y0_values_df.to_dict('index').items())]

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

    def learn_and_run_best_parameter(self):

        print('¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬ self.rules ¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬')
        print(json.dumps(self.rules))
        print('¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬')
        # learn parameters
        best_performing_prior_dict = self.__learn_likelihoods()

        # run monte carlo for best parameters
        (simulation_data_df, triggered_rules_df, errors_df) = self.__run_monte_carlo_simulation(nb_of_simulations=300, prior_dict=best_performing_prior_dict)
        self.__post_process_monte_carlo(simulation_data_df, triggered_rules_df, errors_df, best_performing_prior_dict, 300, 0)

        parameters_were_learned = best_performing_prior_dict!={}
        return parameters_were_learned



    def run_single_monte_carlo(self, number_of_entities_to_simulate, prior_dict, parameter_number):
        (simulation_data_df, triggered_rules_df, errors_df) = self.__run_monte_carlo_simulation(nb_of_simulations=number_of_entities_to_simulate, prior_dict=prior_dict)
        parameter_number = self.__post_process_monte_carlo(simulation_data_df, triggered_rules_df, errors_df, prior_dict, number_of_entities_to_simulate, parameter_number)
        return parameter_number



    def salvage_cancelled_simulation(self, run_number):
        # salvage
        best_performing_prior_dict = self.__retrieve_results_from_cancelled_simulation(run_number)

        # run monte carlo for best parameters
        # (simulation_data_df, triggered_rules_df, errors_df) = self.__run_monte_carlo_simulation(nb_of_simulations=300, prior_dict=best_performing_prior_dict)
        # self.__post_process_monte_carlo(simulation_data_df, triggered_rules_df, errors_df, best_performing_prior_dict, 300, 0)
        # success = best_performing_prior_dict!={}

        simulation_model_record = Simulation_model.objects.get(id=self.simulation_id)
        simulation_model_record.not_used_rules = {}
        simulation_model_record.save()

        success = True
        return success












# ==========================================================================================
#     _                           
#    | |                          
#    | |     ___  __ _ _ __ _ __  
#    | |    / _ \/ _` | '__| '_ \ 
#    | |___|  __/ (_| | |  | | | |
#    |______\___|\__,_|_|  |_| |_|                             
# 
# ==========================================================================================


    def __learn_likelihoods(self):
        print('=======  learn_likelihoods  =======')
        
        # PART 1 - Run the Simulation & populate all_priors_df
        self.currently_running_learn_likelihoods = True
        batch_size = len(self.df)


        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Learning parameters - simulation:', "current_number": 0, "total_number": self.nb_of_tested_parameters * len(self.df)}))


        print('learn likelihoods 1')
        all_priors_df = pd.DataFrame()
        self.nb_of_sim_in_which_rule_was_used = 0
        all_priors_df['error'] = [np.nan] * self.nb_of_tested_parameters
        to_be_learned_priors_exist = False
        for rule in self.rules:
            print('rule %s:  learn_posterior=%s ;has_probability_1=%s; used_parameter_ids=%s' % (rule['id'], rule['learn_posterior'], rule['has_probability_1'], rule['used_parameter_ids']))
            if rule['learn_posterior']:
                all_priors_df['nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] = [np.nan] * self.nb_of_tested_parameters
                all_priors_df['error_rule' + str(rule['id'])] = [np.nan] * self.nb_of_tested_parameters
                if not rule['has_probability_1']:
                    # all_priors_df['triggerThresholdForRule' + str(rule['id'])] = np.random.uniform(0, 1, self.nb_of_tested_parameters)
                    random_values = np.linspace(0,1,self.nb_of_tested_parameters)
                    np.random.shuffle(random_values)
                    all_priors_df['triggerThresholdForRule' + str(rule['id'])] = random_values
                    to_be_learned_priors_exist = True
                for used_parameter_id in rule['used_parameter_ids']:
                    # all_priors_df['param' + str(used_parameter_id)] = np.random.uniform(rule['parameters'][used_parameter_id]['min_value'], rule['parameters'][used_parameter_id]['max_value'], self.nb_of_tested_parameters)
                    random_values = np.linspace(rule['parameters'][used_parameter_id]['min_value'], rule['parameters'][used_parameter_id]['max_value'], self.nb_of_tested_parameters)
                    np.random.shuffle(random_values)
                    all_priors_df['param' + str(used_parameter_id)] = random_values
                    to_be_learned_priors_exist = True
        

        print('learn likelihoods 2 - ' + str(to_be_learned_priors_exist))
        if to_be_learned_priors_exist:
            print('learn likelihoods 3 - ' + str(self.run_locally))
            if self.run_locally:
                # =================  Simulation Loop  ==========================
                for batch_number in range(self.nb_of_tested_parameters):
                    print('learn_likelihoods (%s/%s)' % (batch_number+1, self.nb_of_tested_parameters))

                    with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
                        progress_tracking_file.write(json.dumps({"text": 'Learning parameters - simulation: ', "current_number": batch_number * len(self.df), "total_number": self.nb_of_tested_parameters * len(self.df)}))

                    priors_dict = all_priors_df.loc[batch_number,:].to_dict()

                    y0_values_in_simulation = self.likelihood_learning_simulator(self.df, self.rules, priors_dict, batch_size)
                    errors_dict = self.n_dimensional_distance(y0_values_in_simulation, self.y0_values) 
                    all_priors_df.loc[batch_number, 'error'] = errors_dict['error']
                    for rule in self.rules:
                        if rule['learn_posterior']:
                            all_priors_df.loc[batch_number, 'nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] = errors_dict[rule['id']]['nb_of_sim_in_which_rule_was_used']
                            all_priors_df.loc[batch_number,'error_rule' + str(rule['id'])] = errors_dict[rule['id']]['error'] 
                # ==============================================================
            else:
                for batch_number in range(self.nb_of_tested_parameters):
                    print('posting batch %s : %s' % (batch_number, str(all_priors_df.loc[batch_number,:].to_dict())))
                    print('simulation_id: ' + str(len(str( self.simulation_id))) + '; run_number: ' + str(len(str( self.run_number))) + '; batch_number: ' +  str(len(str(batch_number))) + '; rules: ' +  str(len(str( self.rules ))) + '; priors_dict: ' + str(len(str( all_priors_df.loc[batch_number,:].to_dict()))) + '; batch_size: ' + str(len(str(batch_size ))) + '; is_timeseries_analysis :' +  str(len(str(self.is_timeseries_analysis))) + '; times: ' + str(len(str(self.times))) + '; timestep_size: ' + str(len(str( self.timestep_size))) + '; y0_columns: ' + str(len(str(self.y0_columns))) + '; parameter_columns: '  + str(len(str(self.parameter_columns))) + '; y0_column_dt: ' + str(len(str( self.y0_column_dt))) + '; error_threshold: ' + str(len(str( self.error_threshold))))
                    

                    simulation_parameters = {'y_value_attributes': self.y_value_attributes, 'simulation_id':  self.simulation_id, 'run_number':  self.run_number, 'batch_number': batch_number, 'rules': self.rules , 'priors_dict':  all_priors_df.loc[batch_number,:].to_dict(), 'batch_size': batch_size , 'is_timeseries_analysis': self.is_timeseries_analysis, 'times': self.times, 'timestep_size':  self.timestep_size, 'y0_columns': self.y0_columns, 'parameter_columns':  self.parameter_columns, 'y0_column_dt':  self.y0_column_dt, 'error_threshold':  self.error_threshold}
                    print(simulation_parameters.keys())
                    sqs = boto3.client('sqs', region_name='eu-central-1')
                    queue_url = 'https://sqs.eu-central-1.amazonaws.com/662304246363/Treeofknowledge-queue'
                    response = sqs.send_message(QueueUrl= queue_url, MessageBody=json.dumps(simulation_parameters))

                result_checking_start_time = time.time()
                maximal_execution_time = max(self.nb_of_tested_parameters * 15, 300)
                connection = psycopg2.connect(user="dbadmin", password="rUWFidoMnk0SulVl4u9C", host="aa1pbfgh471h051.cee9izytbdnd.eu-central-1.rds.amazonaws.com", port="5432", database="ebdb")
                cursor = connection.cursor()
                all_simulation_results = []
                while (time.time() - result_checking_start_time < maximal_execution_time):

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
                    all_priors_df.loc[row['batch_number'], 'error'] = simulation_results['error']
                    for rule in self.rules:
                        if rule['learn_posterior']:
                            all_priors_df.loc[row['batch_number'], 'nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] = simulation_results['nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] 
                            all_priors_df.loc[row['batch_number'],'error_rule' + str(rule['id'])] = simulation_results['error_rule' + str(rule['id'])]


            # ---------------  Save Learn_parameters_result  ----------------------
            all_priors_df['nb_of_simulations'] = len(self.df)
            all_priors_df = all_priors_df.sort_values('error')
            all_priors_df.index = range(len(all_priors_df))

            learned_rules = {}
            for object_number in self.objects_dict.keys():
                learned_rules[object_number] = {}
                for attribute_id in self.objects_dict[object_number]['object_rules'].keys():
                    learned_rules[object_number][attribute_id] = {}
                    for rule_id in self.objects_dict[object_number]['object_rules'][attribute_id].keys():
                        if self.objects_dict[object_number]['object_rules'][attribute_id][rule_id]['learn_posterior']:
                            learned_rules[object_number][attribute_id][rule_id] = True
                        else:
                            learned_rules[object_number][attribute_id][rule_id] = False


            learn_parameters_result = Learn_parameters_result(simulation_id=self.simulation_id, execution_order_id=self.execution_order_id, run_number=self.run_number, all_priors_df=json.dumps(all_priors_df.to_dict(orient='index')), learned_rules=json.dumps(learned_rules))
            learn_parameters_result.save()


            # -----------------  best_performing_prior_dict  -------------------------
            best_performing_prior_dict = {}
            for rule_number, rule in enumerate(self.rules):
                if rule['learn_posterior']:
                    best_performing_prior_dict[str(rule['id'])] = {}
                    if not rule['has_probability_1']:
                        best_performing_prior_dict[str(rule['id'])]['probability'] = all_priors_df.loc[0,'triggerThresholdForRule'+ str(rule['id'])]
                    for used_parameter_id in rule['used_parameter_ids']:
                        best_performing_prior_dict[str(rule['id'])][str(used_parameter_id)] = all_priors_df.loc[0,'param' + str(used_parameter_id)]

            return best_performing_prior_dict

        else:
            return {}
            




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



        # df['null'] = np.nan
        # if self.is_timeseries_analysis: 
        #     df['delta_t'] = self.timestep_size
        # else: 
        #     df[self.y0_columns] = None


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



                # --------  new_values  --------
                if rule['effect_is_calculation']: 
                    if 'sums' in rule:
                        for sum_number in rule['sums'].keys():
                            df['sum' + str(sum_number)] = 0
                            for sum_term in rule['sums'][sum_number]:
                                df['sum' + str(sum_number)] += pd.eval(sum_term).fillna(0)

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
                    new_values = pd.Series([rule['effect_exec']] * batch_size)



                # --------  used rules  --------
                if rule['learn_posterior']:
                    rule_was_used_this_period = condition_satisfying_rows & new_values.notnull()
                    rule['rule_was_used_in_simulation'] = rule['rule_was_used_in_simulation'] | rule_was_used_this_period


                # --------  Apply the Change (new_values)  --------
                satisfying_rows[satisfying_rows.isna()] = False
                new_values[np.logical_not(satisfying_rows)] = df.loc[np.logical_not(satisfying_rows),rule['column_to_change']]
                if rule['effect_exec'] != 'df.null':
                    new_values[new_values.isna()] = df.loc[new_values.isna(),rule['column_to_change']]
                df[rule['column_to_change']] = new_values
                # if rule['id'] in [98, 110]:
                    # print('period'+ str(period) +' rule'+ str(rule['id']) +' condition -------------------------------------------')
                    # print('rule[\'condition_exec\'] = ' + rule['condition_exec'])
                    # print(str(list(df['obj1attr226'])))
                    # print(str(list(satisfying_rows)))
                    # print(rule['column_to_change'])
                    # print(str(list(new_values)))
                    # print(str(list(df[rule['column_to_change']])))
                    # print('-------------------------------------------------------------')

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

         





    def __retrieve_results_from_cancelled_simulation(self, run_number):
        connection = psycopg2.connect(user="dbadmin", password="rUWFidoMnk0SulVl4u9C", host="aa1pbfgh471h051.cee9izytbdnd.eu-central-1.rds.amazonaws.com", port="5432", database="ebdb")
        cursor = connection.cursor()
        cursor.execute('''SELECT simulation_id, run_number, batch_number, priors_dict, simulation_results FROM tested_simulation_parameters WHERE simulation_id=%s AND run_number=%s;''' % (self.simulation_id, run_number))
        all_simulation_results = cursor.fetchall() 
        print('checking results - found %s/%s' % (len(all_simulation_results), self.nb_of_tested_parameters))


        all_simulation_results_df = pd.DataFrame(all_simulation_results, columns=['simulation_id', 'run_number', 'batch_number', 'priors_dict', 'simulation_results'])
        priors_dicts = all_simulation_results_df['priors_dict']
        priors_dicts = [json.loads(priors_dict) for priors_dict in priors_dicts]
        priors_df = pd.DataFrame.from_dict(priors_dicts)

        simulation_results = all_simulation_results_df['simulation_results']
        simulation_results = [json.loads(simulation_result) for simulation_result in simulation_results]
        simulation_results_df = pd.DataFrame.from_dict(simulation_results)
        print('[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]')
        print(simulation_results_df)
        print('[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]')

        all_priors_df = pd.DataFrame()
        all_priors_df['error'] = simulation_results_df['error'] 
        for rule in self.rules:
            if rule['learn_posterior']:
                all_priors_df['nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] = simulation_results_df['nb_of_sim_in_which_rule_' + str(rule['id']) + '_was_used'] 
                all_priors_df['error_rule' + str(rule['id'])] = simulation_results_df['error_rule' + str(rule['id'])]

                if not rule['has_probability_1']:
                    all_priors_df['triggerThresholdForRule' + str(rule['id'])] = priors_df['triggerThresholdForRule' + str(rule['id'])]
                for used_parameter_id in rule['used_parameter_ids']:
                    all_priors_df['param' + str(used_parameter_id)] = priors_df['param' + str(used_parameter_id)]



        all_priors_df['nb_of_simulations'] = len(self.df)
        all_priors_df = all_priors_df.sort_values('error')
        all_priors_df.index = range(len(all_priors_df))
        simulation_model_record = Simulation_model.objects.get(id=self.simulation_id)
        simulation_model_record.all_priors_df = json.dumps(all_priors_df.to_dict(orient='index'))
        simulation_model_record.save()

        best_performing_prior_dict = {}
        for rule_number, rule in enumerate(self.rules):
            if rule['learn_posterior']:
                best_performing_prior_dict[str(rule['id'])] = {}
                if not rule['has_probability_1']:
                    best_performing_prior_dict[str(rule['id'])]['probability'] = all_priors_df.loc[0,'triggerThresholdForRule'+ str(rule['id'])]
                for used_parameter_id in rule['used_parameter_ids']:
                    best_performing_prior_dict[str(rule['id'])][str(used_parameter_id)] = all_priors_df.loc[0,'param' + str(used_parameter_id)]

        return best_performing_prior_dict









        
# ==========================================================================================
#    __  __             _          _____           _       
#   |  \/  |           | |        / ____|         | |      
#   | \  / | ___  _ __ | |_ ___  | |     __ _ _ __| | ___  
#   | |\/| |/ _ \| '_ \| __/ _ \ | |    / _` | '__| |/ _ \ 
#   | |  | | (_) | | | | ||  __/ | |___| (_| | |  | | (_) |
#   |_|  |_|\___/|_| |_|\__\___|  \_____\__,_|_|  |_|\___/ 
#                                                                                                                           
# ==========================================================================================






    #  Monte-Carlo  ---------------------------------------------------------------------------------
    def __run_monte_carlo_simulation(self, nb_of_simulations=300, prior_dict={}):
        print('¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬  __run_monte_carlo_simulation   ¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬')
        batch_number = 0 # this is a relict from times when this could run more simulations than len(df), maybe some of this could be refactored sometime

        # if df is longer than nb_of_simulations, shorten it (saves unnecessary computation)
        df_short = self.df[:nb_of_simulations]
        df = df_short.copy()
        y0_values_df_short = self.y0_values_df[:nb_of_simulations]
        y0_values_short = self.y0_values[:nb_of_simulations]

        # print('-----')
        # print('df_short.columns = ' + str(list(df_short.columns)))
        # print('self.y0_columns = ' + str(self.y0_columns))
        # print('-----')
        y0 = np.asarray(df_short[self.y0_columns].copy())
        batch_size = len(y0)


        triggered_rules_df = pd.DataFrame()

        simulation_data_df = df.copy()
        simulation_data_df['initial_state_id'] = df.index
        simulation_data_df['batch_number'] = batch_number
        simulation_data_df['period'] = 0




        for rule in self.rules:
            if rule['changed_var_data_type'] in ['boolean','bool']:
                simulation_data_df[rule['column_to_change']] = simulation_data_df[rule['column_to_change']].astype('object')
            if not rule['has_probability_1']:
                if rule['id'] in prior_dict and 'probability' in prior_dict[str(rule['id'])]:
                    df['triggerThresholdForRule' + str(rule['id'])] = prior_dict[str(rule['id'])]['probability']
                else:
                    df['triggerThresholdForRule' + str(rule['id'])] =  rv_histogram(rule['histogram']).rvs(size=batch_size)
            for used_parameter_id in rule['used_parameter_ids']:

                if str(rule['id']) in prior_dict and str(used_parameter_id) in prior_dict[str(rule['id'])]:
                    df['param' + str(used_parameter_id)] = prior_dict[str(rule['id'])][str(used_parameter_id)]
                else:
                    # print('using histogram - rule%s, parameter%s' % (rule['id'],used_parameter_id))
                    # print(prior_dict)
                    # print(prior_dict[str(rule['id'])])
                    # print(prior_dict[str(rule['id'])][str(used_parameter_id)])
                    # print(rule['parameters'])
                    df['param' + str(used_parameter_id)] = rv_histogram(rule['parameters'][used_parameter_id]['histogram']).rvs(size=batch_size)


        # df['null'] = np.nan
        # if self.is_timeseries_analysis: 
        #     df['delta_t'] = self.timestep_size
        # else: 
        #     df[self.y0_columns] = None

        # if self.is_timeseries_analysis:
            # times = np.arange(self.simulation_start_time + self.timestep_size, self.simulation_end_time, self.timestep_size)
            # times = generally_useful_functions.get_list_of_times(self.simulation_start_time + self.timestep_size, self.simulation_end_time, self.timestep_size)
        # else:
            # times = [self.simulation_start_time, self.simulation_end_time]

        y0_values_in_simulation = pd.DataFrame(index=range(batch_size))
        for period in range(len(self.times[1:])):

            with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
                progress_tracking_file.write(json.dumps({"text": 'Making predictions - simulation: ', "current_number": int(batch_size * (period/len(self.times[1:]))), "total_number": nb_of_simulations}))



            print('period: ' + str(period) + '/' + str(len(self.times[1:])))
            df['randomNumber'] = np.random.random(batch_size)
            print('period: ' + str(period) + ' - 1')
            for rule in self.rules:
                print('period: ' + str(period) + ' - 2,' + str(rule['id']))
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
                print('period: ' + str(period) + ' - 3')
                if rule['effect_is_calculation']:
                    if 'sums' in rule:
                        for sum_number in rule['sums'].keys():
                            df['sum' + str(sum_number)] = 0
                            for sum_term in rule['sums'][sum_number]:
                                df['sum' + str(sum_number)] += pd.eval(sum_term).fillna(0)

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
                print('period: ' + str(period) + ' - 4')
                new_values = all_new_values
                new_values[np.logical_not(satisfying_rows)] = df.loc[np.logical_not(satisfying_rows),rule['column_to_change']]
                if rule['effect_exec'] != 'df.null':
                    new_values[new_values.isna()] = df.loc[new_values.isna(),rule['column_to_change']]
                df[rule['column_to_change']] = new_values


                calculated_values = list(df[rule['column_to_change']])
                errors = np.zeros(len(calculated_values))
                correct_value = ['unknown'] * len(calculated_values)
                if rule['column_to_change'] in self.y0_columns:
                    errors = self.error_of_single_values(df_short, y0_values_df_short, np.array(calculated_values), rule['column_to_change'], period+1)
                    correct_value = list(y0_values_df_short[rule['column_to_change'] + 'period' + str(period+1)])


                triggered_rule_infos_df = pd.DataFrame({'condition_satisfied': condition_satisfying_rows,
                                                        'id':[rule['id']]* batch_size,
                                                        'pt': satisfying_rows,          # pt = probability_triggered
                                                        'tp': trigger_thresholds,       # tp = trigger_probability
                                                        'v': calculated_values,         # v = new_value
                                                        'error':errors})

                print('period: ' + str(period) + ' - 5')
                triggered_rule_infos = triggered_rule_infos_df.to_dict('records')
                triggered_rule_infos = [rule_info if rule_info['condition_satisfied'] else None for rule_info in triggered_rule_infos]
                for i in range(len(triggered_rule_infos)):
                    if triggered_rule_infos[i] is not None:
                            del triggered_rule_infos[i]['condition_satisfied']
                            if np.isnan(triggered_rule_infos[i]['error']):
                                del triggered_rule_infos[i]['error']

                print('period: ' + str(period) + ' - 6')
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


        errors_dict = self.n_dimensional_distance(y0_values_in_simulation.to_dict('records'), y0_values_short)
        errors_df = pd.DataFrame({  'simulation_number': [str(index) + '-' + str(batch_nb) for index, batch_nb in zip(df.index, [batch_number]*len(df))],
                                    'error': errors_dict['all_errors']})


        return (simulation_data_df, triggered_rules_df, errors_df)









    def __post_process_monte_carlo(self, simulation_data_df, triggered_rules_df, errors_df, prior_dict, number_of_simulations, parameter_number):

        print('process_data_1')
        with open(self.progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Preparing results - step: ', "current_number": 1, "total_number": 3}))


       


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
        errors_df.index = errors_df['simulation_number']
        errors['all_errors'] = errors_df['error'].to_dict()
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
        simulation_model_record.not_used_rules = self.not_used_rules
        simulation_model_record.save()


        print('process_data_7')
        if parameter_number is not None:
            monte_carlo_result_record = Monte_carlo_result(simulation_id=self.simulation_id,
                                                        execution_order_id=self.execution_order_id,
                                                        run_number=self.run_number,
                                                        parameter_number=parameter_number,
                                                        is_new_parameter=False,
                                                        prior_dict=json.dumps(prior_dict),
                                                        not_used_rules=self.not_used_rules, 
                                                        triggered_rules=json.dumps(triggered_rules), 
                                                        simulation_data=json.dumps(simulation_data), 
                                                        correct_values=json.dumps(correct_values), 
                                                        errors=json.dumps(errors))
            monte_carlo_result_record.save()



        else:
            monte_carlo_result_record = Monte_carlo_result.objects.filter(simulation_id=self.simulation_id, run_number=self.run_number, is_new_parameter=True).order_by('-parameter_number').first()
            highest_new_parameter_number = 0 if monte_carlo_result_record is None else monte_carlo_result_record.parameter_number
            parameter_number = highest_new_parameter_number + 1
            monte_carlo_result_record = Monte_carlo_result(simulation_id=self.simulation_id,
                                                        execution_order_id=self.execution_order_id,
                                                        run_number=self.run_number,
                                                        parameter_number=parameter_number,
                                                        is_new_parameter=True,
                                                        prior_dict=json.dumps(prior_dict),
                                                        not_used_rules=self.not_used_rules, 
                                                        triggered_rules=json.dumps(triggered_rules), 
                                                        simulation_data=json.dumps(simulation_data), 
                                                        correct_values=json.dumps(correct_values), 
                                                        errors=json.dumps(errors))
            monte_carlo_result_record.save()

        return parameter_number
        












# ===========================================================================================================
#                  _     _ _ _   _                   _   ______                _   _                 
#         /\      | |   | (_) | (_)                 | | |  ____|              | | (_)                
#        /  \   __| | __| |_| |_ _  ___  _ __   __ _| | | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
#       / /\ \ / _` |/ _` | | __| |/ _ \| '_ \ / _` | | |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#      / ____ \ (_| | (_| | | |_| | (_) | | | | (_| | | | |  | |_| | | | | (__| |_| | (_) | | | \__ \
#     /_/    \_\__,_|\__,_|_|\__|_|\___/|_| |_|\__,_|_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
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
      
                    # -------------  version 1  -----------------------
                    # relative_change = np.abs(np.array(u_df[period_column]) - np.array(v_df[period_column.split('period')[0]]))/period_number
                    # normalisation_factor = np.maximum(np.abs(u_df[period_column]),np.abs(v_df[period_column.split('period')[0]]))
                    # normalisation_factor = np.maximum(normalisation_factor, 1)
                    # relative_change = relative_change/normalisation_factor
                    # # relative_change_non_null = np.nan_to_num(relative_change, nan=1.0)  

                    # absolute_change = np.abs(np.array(u_df[period_column]) - np.array(v_df[period_column.split('period')[0]]))
                    # absolute_change = absolute_change/np.abs(np.percentile(absolute_change, 30))
                    # # absolute_change_non_null = np.nan_to_num(absolute_change, nan=1.0) 



                    # -------------  version 2 (works quite well)  -----------------------
                    # residuals = np.abs(np.array(u_df[period_column]) - np.array(v_df[period_column]))
                    # non_null_residuals = residuals[~np.isnan(residuals)]
                    # nth_percentile = np.percentile(non_null_residuals, self.error_threshold*100) if len(non_null_residuals) > 0 else 1# whereby n is the error_threshold. It therefore automatically adapts to the senistivity...
                    # error_divisor = nth_percentile if nth_percentile != 0 else 1
                    # error_in_error_range =  residuals/error_divisor
                    # error_in_error_range = np.log(1 + error_in_error_range)
                    # error_in_error_range_non_null = np.nan_to_num(error_in_error_range, nan=0)  
                    # error_in_error_range_non_null = np.minimum(error_in_error_range_non_null, 1)

                    # true_change_factor = (np.array(v_df[period_column])/np.array(v_df[period_column.split('period')[0]]))
                    # true_change_factor_per_period = np.power(true_change_factor, (1/period_number))
                    # simulated_change_factor = (np.array(u_df[period_column])/np.array(v_df[period_column.split('period')[0]]))
                    # simulated_change_factor_per_period = np.power(simulated_change_factor, (1/period_number))
                    # error_of_value_change = np.abs(simulated_change_factor_per_period - true_change_factor_per_period) 
                    # error_of_value_change = np.log(1 + error_of_value_change)
                    # error_of_value_change_non_null = np.nan_to_num(error_of_value_change, nan=0)  
                    # error_of_value_change_non_null = np.minimum(error_of_value_change_non_null, 1)

                    # error = 0.5*np.minimum(error_in_error_range_non_null,error_of_value_change_non_null) + 0.25*np.sqrt(error_in_error_range_non_null) + 0.25*np.sqrt(error_of_value_change_non_null)


                    # -------------  version 3 - RANKS (excellent, only not good for overall score) -----------------------
                    # residual_rank = np.abs(np.array(u_df[period_column]) - np.array(v_df[period_column]))
                    # not_null_index = pd.notnull(residual_rank)
                    # residual_rank[not_null_index] = rankdata(residual_rank[not_null_index])


                    # true_change_factor = (np.array(v_df[period_column])/np.array(v_df[period_column.split('period')[0]]))
                    # true_change_factor_per_period = np.power(true_change_factor, (1/period_number))
                    # simulated_change_factor = (np.array(u_df[period_column])/np.array(v_df[period_column.split('period')[0]]))
                    # simulated_change_factor_per_period = np.power(simulated_change_factor, (1/period_number))
                    # value_change_rank = np.abs(simulated_change_factor_per_period - true_change_factor_per_period) 
                    # not_null_index = pd.notnull(value_change_rank)
                    # value_change_rank[not_null_index] = rankdata(value_change_rank[not_null_index])

                    # both_ranks = np.array([residual_rank, value_change_rank])
                    # error = np.nanmin(both_ranks, axis=0) + np.nanmax(both_ranks, axis=0) 




                    # -------------  version 4 -----------------------
                    residuals = np.abs(np.array(u_df[period_column]) - np.array(v_df[period_column]))
                    non_null_residuals = residuals[~np.isnan(residuals)]
                    nth_percentile = np.percentile(non_null_residuals, self.error_threshold*100) if len(non_null_residuals) > 0 else 1# whereby n is the error_threshold. It therefore automatically adapts to the senistivity...
                    error_divisor = nth_percentile if nth_percentile != 0 else 1
                    error_in_error_range =  residuals/error_divisor
                    error_in_error_range = np.sqrt(error_in_error_range)

                    true_change_factor = (np.array(v_df[period_column])/np.array(v_df[period_column.split('period')[0]]))
                    true_change_factor_per_period = np.power(true_change_factor, (1/period_number))
                    simulated_change_factor = (np.array(u_df[period_column])/np.array(v_df[period_column.split('period')[0]]))
                    simulated_change_factor_per_period = np.power(simulated_change_factor, (1/period_number))
                    error_of_value_change = np.abs(simulated_change_factor_per_period - true_change_factor_per_period) 
                    error_of_value_change = np.sqrt(error_of_value_change)

                    both_errors = np.array([error_in_error_range, error_of_value_change])
                    error = (np.nanmin(both_errors, axis=0) + np.nanmax(both_errors, axis=0))/ 2
                    error = 1 - np.exp(-2*error)
                    null_value_places = np.isnan(error)
                    error[null_value_places] = 0
                    dimensionality += 1 - null_value_places.astype('int')
                    total_error += error 

                    # generally_useful_functions.log(v_df[period_column], 'v_df[period_column]')
                    # generally_useful_functions.log(u_df[period_column], 'u_df[period_column]')
                    # generally_useful_functions.log(v_df[period_column.split('period')[0]], 'v_df[period_column.split(\'period\')[0]]')
                    # generally_useful_functions.log(error, 'error')


        non_validated_rows = dimensionality == 0
        dimensionality = np.maximum(dimensionality, [1]*len(u))
        error = total_error/dimensionality
        error[non_validated_rows] = 1



        if len(error)==0:
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print('v_df=' + json.dumps(list(u)))
            print('u_df=' + json.dumps(list(v)))
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        errors_dict = {'all_errors':error, 'error': error.mean()}
        if self.currently_running_learn_likelihoods:
            for rule in self.rules:
                if rule['learn_posterior'] and 'rule_used_in_simulation_' + str(rule['id']) in u_df.columns:
                    rule_used_in_simulation = u_df['rule_used_in_simulation_' + str(rule['id'])]
                    errors_dict[rule['id']] = {'error': error[rule_used_in_simulation].mean(), 'nb_of_sim_in_which_rule_was_used': rule_used_in_simulation.sum()}

        return errors_dict



    def error_of_single_values(self, df, y0_values_df, calculated_values, column_name, period):      
        initial_values = np.array(df[column_name])
        correct_values = np.array(y0_values_df[column_name + 'period' + str(period)])


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
            # errors = np.minimum(errors, 1)

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




