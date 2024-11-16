####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (C) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, February 2021
#####################################################################

from collection.models import Object_types, Attribute, Object, Data_point, Simulation_model
import numpy as np
import pandas as pd
import datetime
from collection.functions import generally_useful_functions, get_from_db
import random
from django.db.models import Count
from django.db import connection
import json
import sqlite3
import itertools
import math
import os
import networkx as nx
import pdb


def find_matching_entities(match_attributes, match_values):

    # print('=============================================================================')
    # print(match_attributes)
    # print('-------------------')
    # print(match_values)
    # print('=============================================================================')
    with connection.cursor() as cursor:

        # create table_to_match   ----------------------------------------
        cursor.execute('DROP TABLE IF EXISTS table_to_match')
        create_match_table = '''
            CREATE TEMPORARY TABLE table_to_match (
               row_number INT,
               attribute_id TEXT, 
               value_as_string TEXT
            ); ''' 
        cursor.execute(create_match_table)


        # insert into table_to_match   ----------------------------------------
        number_of_rows = len(match_values[0])
        row_number_column = list(range(number_of_rows))*len(match_attributes)
        attribute_id_column = []
        value_as_string_column = []
        for column_number, match_attribute in enumerate(match_attributes):
            attribute_id_column.extend( [str(match_attribute['attribute_id'])] * number_of_rows)
            value_as_string_column.extend([str(value) for value in match_values[column_number]])


        table_rows = list(map(list, zip(*[row_number_column, attribute_id_column, value_as_string_column])))


        number_of_chunks =  math.ceil(number_of_rows / 100)
        for chunk_index in range(number_of_chunks):

            rows_to_insert = table_rows[chunk_index*100: chunk_index*100 + 100]
            insert_statement = '''
                INSERT INTO table_to_match (row_number, attribute_id, value_as_string) 
                VALUES ''' 
            insert_statement += ','.join(['(%s, %s, %s)']*len(rows_to_insert))
            cursor.execute(insert_statement, list(itertools.chain.from_iterable(rows_to_insert)))
            # insert_statement = '''
            #     INSERT INTO table_to_match (row_number, attribute_id, value_as_string) 
            #     VALUES ''' 
            # insert_statement += ','.join(["(%s, '%s', '%s')"]*len(rows_to_insert))
            # flattened_list = [y for x in rows_to_insert for y in x]
            # flattened_list = [el.replace("'","''") if isinstance(el,str) else el for el in flattened_list]
            # cursor.execute(insert_statement % tuple(flattened_list))



        # match table_to_match with collection_data_point   ----------------------------------------
        matched_data_points_string = """
            CREATE TEMPORARY TABLE matched_data_points AS
                SELECT  row_number, 
                        object_id, 
                        dp.attribute_id, 
                        dp.value_as_string, 
                        '"' || dp.attribute_id || '":"' || dp.value_as_string || '"' AS dictionary_element,
                        MAX(data_quality) AS data_quality
                FROM table_to_match AS ttm
                LEFT JOIN collection_data_point AS dp
                ON ttm.attribute_id = dp.attribute_id AND 
                   ttm.value_as_string = dp.value_as_string 
                WHERE ttm.value_as_string != 'None'
                GROUP BY row_number, object_id, dp.attribute_id, dp.value_as_string ; 
        """
        cursor.execute(matched_data_points_string)



        # ----------  POSTGRES VS. SQLITE  ----------
        # group_concat (sqlite) vs. string_agg (postgres)

        # if 'DATABASE_URL' in dict(os.environ).keys() and dict(os.environ)['DATABASE_URL'][:8]!='postgres':
        if 'IS_USING_SQLITE_DB' in dict(os.environ).keys():
            # SQLITE...
            matched_objects_string = """
                CREATE TEMPORARY TABLE matched_objects AS
                    SELECT  row_number, 
                            object_id, 
                            '{"object_id":' || object_id || ', ' || group_concat(dictionary_element) || '}' AS object_dict, 
                            COUNT(*) AS number_of_attributes_found,
                            SUM(data_quality) AS data_quality,
                            RANK () OVER (PARTITION BY row_number ORDER BY data_quality DESC) AS match_number
                    FROM matched_data_points
                    GROUP BY row_number, object_id;
            """
            cursor.execute(matched_objects_string)


            matched_rows_string = """
                CREATE TEMPORARY TABLE matched_rows AS
                    SELECT 
                        row_number, 
                        '[' || group_concat(object_dict) || ']'  AS matching_objects_json
                    FROM matched_objects
                    WHERE number_of_attributes_found > 0
                      AND match_number <=3
                    GROUP BY row_number;
            """
            cursor.execute(matched_rows_string)


            row_number_string = """
                CREATE TEMPORARY TABLE row_number AS
                    SELECT DISTINCT row_number
                    FROM table_to_match  
                    ORDER BY row_number;
            """
            cursor.execute(row_number_string)


            get_matching_objects_json = """
                SELECT '[' || group_concat(matching_objects_json) || ']' AS matching_objects_json
                FROM (
                    SELECT  COALESCE(mr.matching_objects_json, '[]') AS matching_objects_json
                    FROM row_number AS rn
                    LEFT JOIN matched_rows  AS mr
                    ON rn.row_number = mr.row_number
                    ORDER BY rn.row_number
                );
            """

            result = cursor.execute(get_matching_objects_json)
            matching_objects_entire_list_string = list(result)[0][0]
            return matching_objects_entire_list_string



        else:    
            # POSTGES...  
            matched_objects_string = """
                CREATE TEMPORARY TABLE matched_objects AS
                    SELECT  row_number, 
                            object_id, 
                            '{"object_id":' || object_id || ', ' || string_agg(dictionary_element, ',') || '}' AS object_dict, 
                            COUNT(*) AS number_of_attributes_found,
                            SUM(data_quality) AS data_quality,
                            RANK () OVER (PARTITION BY row_number ORDER BY SUM(data_quality) DESC) AS match_number
                    FROM matched_data_points
                    GROUP BY row_number, object_id;
            """
            cursor.execute(matched_objects_string)



            matched_rows_string = """
                CREATE TEMPORARY TABLE matched_rows AS
                    SELECT 
                        row_number, 
                        '[' || string_agg(object_dict, ',') || ']'  AS matching_objects_json
                    FROM matched_objects
                    WHERE number_of_attributes_found > 0
                      AND match_number <=3
                    GROUP BY row_number;
            """
            cursor.execute(matched_rows_string)


            row_number_string = """
                CREATE TEMPORARY TABLE row_number AS
                    SELECT DISTINCT row_number
                    FROM table_to_match  
                    ORDER BY row_number;
            """
            cursor.execute(row_number_string)


            get_matching_objects_json = """
                SELECT '[' || string_agg(foo.matching_objects_json, ',') || ']' AS matching_objects_json
                FROM (
                    SELECT  COALESCE(mr.matching_objects_json, '[]') AS matching_objects_json
                    FROM row_number AS rn
                    LEFT JOIN matched_rows  AS mr
                    ON rn.row_number = mr.row_number
                    ORDER BY rn.row_number
                ) foo;
            """

            result = cursor.execute(get_matching_objects_json)
            matching_objects_entire_list_string = cursor.fetchall()[0][0]
            return matching_objects_entire_list_string










# this function should be extended to also find fuzzy matches and suggest them in the format_violation_text
def find_single_entity(relation_id, attribute_id, value):
    print('==============  find_single_entity  =====================')
    print(str(relation_id) + ',' + str(attribute_id) + ',' + str(value))
    first_relation_object_type = Attribute.objects.get(id=relation_id).first_relation_object_type
    print(str(first_relation_object_type))
    list_of_parent_objects = get_from_db.get_list_of_parent_objects(first_relation_object_type)
    print(str(list_of_parent_objects))
    list_of_parent_object_ids = [parent_obj['id'] for parent_obj in list_of_parent_objects]
    print(str(list_of_parent_object_ids))
    list_of_object_ids = list(Object.objects.filter(object_type_id__in=list_of_parent_object_ids).values_list('id'))
    print(str(list_of_object_ids))
    list_of_object_ids = [el[0] for el in list_of_object_ids]
    print(str(list_of_object_ids))
    matching_objects_list = list(Data_point.objects.filter(object_id__in=list_of_object_ids, attribute_id=attribute_id, value_as_string=value).values())
    print(str(matching_objects_list))
    if len(matching_objects_list)>0:
        return list(Data_point.objects.filter(object_id__in=list_of_object_ids, attribute_id=attribute_id, value_as_string=value).values())[0]['object_id']
    else:
        return None









def get_data_points(object_type_id, filter_facts, specified_start_time, specified_end_time):

    # basic filtering: object_type and specified time range
    child_object_types = get_from_db.get_list_of_child_objects(object_type_id)
    child_object_ids = [el['id'] for el in child_object_types]


    with connection.cursor() as cursor:
        query_string = "SELECT DISTINCT id FROM collection_object WHERE object_type_id IN (%s);" % (", ".join("'{0}'".format(object_type_id) for object_type_id in child_object_ids))
        cursor.execute(query_string)
        object_ids = [entry[0] for entry in cursor.fetchall()]

    

    # object_ids = list(Object.objects.filter(object_type_id__in=child_object_ids).values_list('id', flat=True))
    broad_table_df = filter_and_make_df_from_datapoints(object_type_id, object_ids, filter_facts, specified_start_time, specified_end_time)   


    # prepare response
    if broad_table_df is not None:

        # for response: list of the tables' attributes sorted with best populated first
        table_attributes = []
        sorted_attribute_ids = broad_table_df.notnull().sum().sort_values(ascending=False).index
        sorted_attribute_ids = [int(attribute_id) for attribute_id in list(sorted_attribute_ids) if attribute_id.isdigit()]

        for attribute_id in sorted_attribute_ids:
            attribute_record = Attribute.objects.get(id=attribute_id)
            table_attributes.append({'attribute_id':attribute_id, 'attribute_name':attribute_record.name, 'attribute_data_type':attribute_record.data_type, 'attribute_population': int(broad_table_df[str(attribute_id)].count())})
            

        # sort broad_table_df -  the best-populated entities to the top
        broad_table_df = broad_table_df.loc[broad_table_df.isnull().sum(1).sort_values().index]


        response = {}
        response['table_body'] = broad_table_df.to_dict('list')
        response['table_attributes'] = table_attributes
        response['number_of_entities_found'] = len(broad_table_df)
    else: 
        response = {}
        response['table_body'] = {}
        response['table_attributes'] = []
        response['number_of_entities_found'] = 0

    return response





# used in edit_model.html
def get_data_from_random_object(object_type_id, filter_facts, specified_start_time, specified_end_time): 
    
        # basic filtering: object_type and specified time range
    child_object_types = get_from_db.get_list_of_child_objects(object_type_id)
    child_object_ids = [el['id'] for el in child_object_types]

    with connection.cursor() as cursor:
        query_string = 'SELECT DISTINCT id FROM collection_object WHERE object_type_id IN (%s);' % (', '.join('"{0}"'.format(object_type_id) for object_type_id in child_object_ids))
        cursor.execute(query_string)
        object_ids = [entry[0] for entry in cursor.fetchall()]

    broad_table_df = filter_and_make_df_from_datapoints(object_type_id, object_ids, filter_facts, specified_start_time, specified_end_time)   

    if broad_table_df is not None:

        broad_table_df.index = range(len(broad_table_df))
        found_object_ids = list(broad_table_df['object_id'])
        chosen_object_id = random.choice(found_object_ids)
        object_record = broad_table_df[broad_table_df['object_id'] == chosen_object_id ]

        attribute_values = {}
        attribute_ids = [int(col) for col in object_record.columns if col not in ['object_id','time']]

        for attribute_id in attribute_ids:
            attribute_record = Attribute.objects.get(id=attribute_id)
            attribute_values[attribute_id] = {  'attribute_value': broad_table_df[str(attribute_id)].iloc[0], 
                                                'attribute_name':attribute_record.name, 
                                                'attribute_data_type':attribute_record.data_type, 
                                                'attribute_rule': None}

    else: 
        chosen_object_id = None
        attribute_values = {}

    return (chosen_object_id, attribute_values)




def get_data_from_random_related_object(simulation_id, objects_dict, environment_start_time, environment_end_time, max_number_of_instances):
    objects_data = {}

    print('~~~~~~~~~~~~~~~  get_data_from_random_related_object  ~~~~~~~~~~~~~~~~~~~~~~')
    object_numbers = list(objects_dict.keys())
    # merged_object_data_tables = get_data_from_related_objects(objects_dict, environment_start_time, environment_end_time, simulation_id)
    merged_object_data_tables = get_data_from_related_objects__single_timestep(objects_dict, environment_start_time, environment_end_time, simulation_id, max_number_of_instances)
    print('len(merged_object_data_tables): ' + str(len(merged_object_data_tables)))

    if len(merged_object_data_tables) > 0:

        # Sort Columns - top = biggest_number_of_non_nulls - attribute_id
        print('sorting columns 1')
        non_object_id_columns = [column for column in list(merged_object_data_tables.columns) if (len(column.split('attr'))>1) and (column.split('attr')[1] not in ['object_id','time'])]
        columns_df = merged_object_data_tables[non_object_id_columns].notnull().sum()
        attribute_ids = [int(column.split('attr')[1]) for column in non_object_id_columns]
        columns_df = columns_df - pd.Series(attribute_ids, index=non_object_id_columns)

        print('sorting columns 2')
        sorted_column_names = columns_df.sort_values(ascending=False).index
        sorted_attribute_ids = [column.split('attr')[1] for column in sorted_column_names]
        sorted_attribute_ids = list(dict.fromkeys(sorted_attribute_ids))  # remove duplicate attribute_ids
        objects_data['sorted_attribute_ids'] = sorted_attribute_ids

        

        # Sort Rows -  the best-populated entities to the top
        print('sorting rows')
        merged_object_data_tables = merged_object_data_tables.loc[merged_object_data_tables.isnull().sum(1).sort_values().index]
        merged_object_data_tables.index = range(len(merged_object_data_tables))    
        chosen_row = 0 # chose top row
        # chosen_row = random.choice(merged_object_data_tables.index)  # chose random row
    

        # prepare return values (all_attribute_values)
        print('preparing return values')
        all_attribute_values = {}
        for object_number in object_numbers:
            all_attribute_values[object_number] = { 'object_id': int(merged_object_data_tables.loc[chosen_row, 'obj' + str(object_number) + 'attrobject_id']),
                                                    'object_attributes':{} }
            object_columns = [column for column in merged_object_data_tables.columns if (column.split('attr')[0][3:]==str(object_number)) and (column.split('attr')[1] not in ['object_id','time'])]

            for object_column in object_columns:
                attribute_id = object_column.split('attr')[1]
                attribute_record = Attribute.objects.get(id=attribute_id)
                attribute_value = merged_object_data_tables['obj' + str(object_number) + 'attr' + str(attribute_id)].iloc[chosen_row]
                if isinstance(attribute_value, float) and np.isnan(attribute_value):
                    attribute_value = None
                if attribute_record.data_type in ['int', 'relation'] and attribute_value is not None:
                    attribute_value = int(attribute_value)
                all_attribute_values[object_number]['object_attributes'][attribute_id] = {  'attribute_value': attribute_value, 
                                                                                            'attribute_name':attribute_record.name, 
                                                                                            'attribute_data_type':attribute_record.data_type, 
                                                                                            'attribute_rule': None}
        objects_data['all_attribute_values'] = all_attribute_values

    else: 
        objects_data['all_attribute_values'] = {object_number:{'object_attributes':{}} for object_number in object_numbers}
        objects_data['sorted_attribute_ids'] = []
        
    print('-- end get_data_from_random_related_object --')
    return objects_data








def get_data_from_related_objects__single_timestep(objects_dict, valid_time_start, valid_time_end, simulation_id, max_number_of_instances, y0_columns=None):
    print('~~~~~~~~~~~~~~~~~  get_data_from_related_objects__single_timestep  ~~~~~~~~~~~~~~~~~')
    print('valid_time_start:' + str(valid_time_start))
    print('valid_time_end:' + str(valid_time_end))

    sqlite_database = 'IS_USING_SQLITE_DB' in dict(os.environ).keys()
    object_numbers = list(objects_dict.keys())
    data_querying_info = {"timestamps":{}, "table_sizes":{}, "relation_sizes":{}}
    # ==================================================================================================
    # PART 1: for every object get a table with the object id that fulfill the internal criteria
    # ==================================================================================================
    print('part1')
    with connection.cursor() as cursor:
        for object_number in object_numbers:

            object_type_id = objects_dict[object_number]['object_type_id']
            relation_ids = [relation['attribute_id'] for relation in objects_dict[object_number]['object_relations']]
            filter_facts  = objects_dict[object_number]['object_filter_facts']

            child_object_types = get_from_db.get_list_of_child_objects(object_type_id)
            child_object_type_ids = [el['id'] for el in child_object_types]
            child_object_types_string = ", ".join("'{0}'".format(object_type_id) for object_type_id in child_object_type_ids)



            # 1.1 create table with object_ids and potentially relation_ids
            cursor.execute('DROP TABLE IF EXISTS object_%s' % str(object_number))


            if len(relation_ids) == 0:                
                sql_string1 = """CREATE TEMPORARY TABLE object_%s AS
                                    SELECT inner_query.* 
                                    FROM (
                                        SELECT DISTINCT object_id AS obj%sattrobject_id
                                        FROM collection_data_point
                                        WHERE valid_time_start <= %s
                                          AND valid_time_end >= %s
                                          AND object_id IN (
                                                            SELECT DISTINCT id 
                                                            FROM collection_object 
                                                            WHERE object_type_id IN (%s)
                                                            )
                                """ % (str(object_number), str(object_number), str(valid_time_end), str(valid_time_start), child_object_types_string)

            elif len(relation_ids) == 1:
                sql_string1 = """CREATE TEMPORARY TABLE object_%s AS
                                    SELECT inner_query.* 
                                    FROM (
                                        SELECT related_objects.object_id AS obj%sattrobject_id, related_objects.numeric_value AS object_%s_relation_%s
                                        FROM (
                                            SELECT DISTINCT object_id, numeric_value 
                                            FROM collection_data_point
                                            WHERE attribute_id = '%s'
                                              AND valid_time_start <= %s
                                              AND valid_time_end >= %s
                                              AND object_id IN (
                                                                SELECT DISTINCT id 
                                                                FROM collection_object 
                                                                WHERE object_type_id IN (%s)
                                                                )
                                            ) related_objects
                                """ % (str(object_number), str(object_number), str(object_number), str(relation_ids[0]), str(relation_ids[0]), str(valid_time_end), str(valid_time_start), child_object_types_string)

            else: 
                cursor.execute('DROP TABLE IF EXISTS object_%s__with_missing_relations' % str(object_number))
                sql_string1 = """CREATE TEMPORARY TABLE object_%s__with_missing_relations AS
                                    SELECT inner_query.* 
                                    FROM (
                                        SELECT related_objects.object_id, related_objects.numeric_value AS object_%s_relation_%s
                                        FROM (
                                            SELECT DISTINCT object_id, numeric_value 
                                            FROM collection_data_point
                                            WHERE attribute_id = '%s'
                                              AND valid_time_start <= %s
                                              AND valid_time_end >= %s
                                              AND object_id IN (
                                                                SELECT DISTINCT id 
                                                                FROM collection_object 
                                                                WHERE object_type_id IN (%s)
                                                                )
                                            ) related_objects 
                                """ % (str(object_number), str(object_number), str(relation_ids[0]), str(relation_ids[0]), str(valid_time_end), str(valid_time_start), child_object_types_string)



            if len(filter_facts)>0 or y0_columns is not None:
                if len(relation_ids) > 0:
                    sql_string1    += ''' INNER JOIN ( '''
                else: 
                    sql_string1    += ''' INTERSECT '''



                for fact_index, filter_fact in enumerate(filter_facts):
                    if fact_index > 0:
                        sql_string1 += ''' INTERSECT '''

                    sql_string1    += '''   SELECT DISTINCT object_id
                                            FROM collection_data_point
                                            WHERE 
                    '''
                    if filter_fact['operation'] == '=':     
                        sql_string1 +=            "attribute_id = '%s' AND string_value = '%s' AND valid_time_start <= %s AND valid_time_end >= %s " % (filter_fact['attribute_id'], filter_fact['value'], str(valid_time_end), str(valid_time_start))
                    elif filter_fact['operation'] == '>':
                        sql_string1 +=            "attribute_id = '%s' AND numeric_value > %s AND valid_time_start <= %s AND valid_time_end >= %s " % (filter_fact['attribute_id'], filter_fact['value'], str(valid_time_end), str(valid_time_start))
                    elif filter_fact['operation'] == '<':
                        sql_string1 +=            "attribute_id = '%s' AND numeric_value < %s AND valid_time_start <= %s AND valid_time_end >= %s " % (filter_fact['attribute_id'], filter_fact['value'], str(valid_time_end), str(valid_time_start))
                    elif filter_fact['operation'] == 'in':
                        values = ['"%s"' % value for value in filter_fact['value']]
                        sql_string1 +=            "attribute_id = '%s' AND string_value IN (%s) AND valid_time_start <= %s AND valid_time_end >= %s " % (filter_fact['attribute_id'], ', '.join(values), str(valid_time_end), str(valid_time_start))



                if y0_columns is not None:
                    attribute_ids = [col.split('attr')[1] for col in y0_columns if 'obj' + str(object_number) + 'attr' in col]
                    for attribute_id in attribute_ids:
                        sql_string1    += '''INTERSECT
                                                SELECT DISTINCT object_id
                                                FROM collection_data_point
                                                WHERE attribute_id = '%s'
                                                  AND valid_time_start <= %s
                                                  AND valid_time_end >= %s
                                            INTERSECT
                                                SELECT DISTINCT object_id
                                                FROM collection_data_point
                                                WHERE attribute_id = '%s'
                                                  AND valid_time_start <= %s
                                                  AND valid_time_end >= %s
                    ''' % (attribute_id, str(valid_time_end), str(valid_time_start), attribute_id, str(valid_time_end), str(valid_time_end))


                if len(relation_ids) > 0:
                    sql_string1    += ''') fact_objects
                                        ON fact_objects.object_id = related_objects.object_id ''' 

                
            print('1.1')    
            sql_string1    += ''' ) as inner_query
                                ORDER BY RANDOM()
                                LIMIT %s;''' % max_number_of_instances
            print(sql_string1)
            cursor.execute(sql_string1)



            # 1.2 add any missing relation_ids to the table
            print('1.2')
            if len(relation_ids) > 1: 
                missing_relations = relation_ids[1:]
                missing_relations_string = ", ".join("relation_table_%s.object_%s_relation_%s" % (relation_id, object_number, relation_id) for relation_id in missing_relations)

                sql_string2 = """CREATE TEMPORARY TABLE object_%s AS
                                    SELECT original_table.object_id AS obj%sattrobject_id, original_table.object_%s_relation_%s, %s
                                    FROM object_%s__with_missing_relations AS original_table

                            """ % (str(object_number), str(object_number), str(object_number), str(relation_ids[0]), missing_relations_string, str(object_number))

                for missing_relation in missing_relations:
                    sql_string2 += """INNER JOIN (
                                            SELECT DISTINCT object_id, numeric_value AS object_%s_relation_%s
                                            FROM collection_data_point
                                            WHERE attribute_id = '%s'
                                              AND valid_time_start >= %s
                                              AND valid_time_end <= %s
                                              AND object_id IN (SELECT DISTINCT object_id FROM object_%s__with_missing_relations)
                                    ) AS relation_table_%s
                                    ON original_table.object_id = relation_table_%s.object_id 
                                """ % (str(object_number), str(missing_relation), str(missing_relation), str(valid_time_start), str(valid_time_end), str(object_number), str(missing_relation), str(missing_relation))
                print(sql_string2)
                cursor.execute(sql_string2)
            

        # 1.3 get data_querying_info['relation_sizes']
        for object_number in object_numbers: 
            print('1.3')
            for relation in objects_dict[object_number]['object_relations']:
                cursor.execute("SELECT DISTINCT object_%s_relation_%s FROM object_%s;" % (object_number, relation['attribute_id'], object_number))
                source = list(cursor.fetchall()[0])

                cursor.execute("SELECT DISTINCT obj%sattrobject_id FROM object_%s;" % (relation['target_object_number'], relation['target_object_number']))
                target = list(cursor.fetchall()[0])

                matched_from_source = len([el for el in source if el in set(target)])
                matched_from_target = len([el for el in target if el in set(source)])
                link_id = str(object_number) + ',' + str(relation['attribute_id']) + ',' + str(relation['target_object_number'])
                data_querying_info['relation_sizes'][link_id] = { 'matched_from_source':matched_from_source,
                                                                'matched_from_target':matched_from_target}


    # ==================================================================================================
    # PART 2: join the object-tables
    # ==================================================================================================   
        print('part2')
        # 2.1 prepare directed graph 
        G = nx.MultiDiGraph()
        node_sizes = {}
        edge_counter = 0
        edge_info = {}

        for object_number in object_numbers: 
            # populate node_sizes    
            # if sqlite_database:
            cursor.execute("SELECT COUNT(DISTINCT obj%sattrobject_id) AS approximate_row_count FROM object_%s;" % (object_number, object_number))
            # else:
            #     cursor.execute("SELECT reltuples AS approximate_row_count FROM pg_class WHERE relname = 'object_%s';")
            objects_table_length = 0
            objects_table_length = cursor.fetchall()[0][0]
            node_sizes[object_number] = objects_table_length
            data_querying_info['table_sizes'][object_number] = {'number_of_objects': objects_table_length, 'number_of_matches': {}} 


            # add nodes and edges (the edge-weight is an id used for storing additional edge_info)
            G.add_node(object_number) 
            for relation in objects_dict[object_number]['object_relations']:
                edge_counter += 1
                G.add_edge(object_number, str(relation['target_object_number']), weight=edge_counter)
                edge_info[edge_counter] = {'related_object_nb': str(relation['target_object_number']), 'relation_column_name': 'object_%s_relation_%s' % (object_number, relation['attribute_id'])}

 


        # 2.2 sequential collapse
        while len(G.edges()) > 0:
            # sort the edges - we always collapse/contract the edge with the lowest score
            edges_with_scores = [(edge[3], (edge[0], edge[1]), (node_sizes[str(edge[0])]*node_sizes[str(edge[1])]), edge[2]) for edge in list(G.edges_iter(data='weight', keys=True))]
            print('sequential collapse: ' + str(edges_with_scores))
            print(str(node_sizes))
            # pattern: [(<weight=edge_number>, (<origin_object_nb>, <target_object_nb>), <edge_score>, <edge_key>), ...]
            edges_with_scores = sorted(edges_with_scores, key=lambda tup: tup[2]) # sort by edge_score
            edge_number = edges_with_scores[0][0]
            origin_object_nb = edges_with_scores[0][1][0]
            target_object_nb = edges_with_scores[0][1][1] 
            edge_key = edges_with_scores[0][3]
            relation_column_name = edge_info[edge_number]['relation_column_name']
            related_object_nb = edge_info[edge_number]['related_object_nb']


            # the target object will be fused into the origin i.e. the target is removed
            # collapsing the tables
            sql_string3_1 = '''ALTER TABLE object_%s RENAME TO object_%s__temp;''' % (origin_object_nb, origin_object_nb)
            cursor.execute(sql_string3_1)
            
            if origin_object_nb == target_object_nb:
                target_table_name = 'object_%s__temp' % target_object_nb
            else:
                target_table_name = 'object_%s' % target_object_nb    

            sql_string3_2 = ''' CREATE TEMPORARY TABLE object_%s AS
                                SELECT origin.*, target.*
                                FROM object_%s__temp AS origin
                                INNER JOIN (
                                    SELECT * FROM %s 
                                ) AS target
                                ON origin.%s = target.obj%sattrobject_id 
                                WHERE origin.obj%sattrobject_id <> target.obj%sattrobject_id 
                                ORDER BY random()
                                LIMIT %s;
                        ''' % (origin_object_nb, origin_object_nb, target_table_name, relation_column_name, related_object_nb, origin_object_nb, related_object_nb, max_number_of_instances)
            cursor.execute(sql_string3_2)
            sql_string3_3 = '''DROP TABLE object_%s__temp;''' % origin_object_nb
            cursor.execute(sql_string3_3)

            #     sql_string3 = '''  
            #             ALTER TABLE object_%s RENAME TO object_%s__temp;
            #             TRUNCATE object_%s;
            #             INSERT INTO object_%s
            #             SELECT * 
            #             FROM object_%s__temp AS origin
            #             INNER JOIN (
            #                 SELECT * FROM object_%s
            #             ) AS target
            #             ON origin.%s = target.object_%s_object_id 
            #             WHERE origin.object_%s_object_id <> target.object_%s_object_id 
            #             LIMIT %s;
            #             ''' % (origin_object_nb, origin_object_nb, origin_object_nb, origin_object_nb, origin_object_nb, target_object_nb, relation_column_name, related_object_nb, origin_object_nb, related_object_nb, max_number_of_instances)
            #     cursor.execute(sql_string3)

            # collapsing the graph
            G.remove_edge(str(origin_object_nb), str(target_object_nb),key=edge_key)
            G = nx.contracted_nodes(G, str(origin_object_nb), str(target_object_nb))

            # updating node_sizes
            cursor.execute("SELECT COUNT(*) FROM object_%s;" % origin_object_nb)
            new_origin_table_length = cursor.fetchall()[0][0]
            node_sizes[str(origin_object_nb)] = new_origin_table_length
            del node_sizes[str(target_object_nb)] 


        # 2.3 cross-join the remaining nodes/tables
        print(node_sizes)
        cursor.execute('''DROP TABLE IF EXISTS object_ids_table;''')
        if len(G.nodes()) > 1:

            select_columns = ", ".join("object_%s.*" % (node_id) for node_id in list(G.nodes()))
            sql_string4 = '''  
                        CREATE TEMPORARY TABLE object_ids_table AS
                            SELECT %s
                            FROM object_%s 
                        ''' % (select_columns, list(G.nodes())[0])
            for node_id in list(G.nodes())[1:]:
                sql_string4 += 'CROSS JOIN object_%s ' % node_id
            sql_string4 += ';' 
            cursor.execute(sql_string4)
        else:
            cursor.execute('ALTER TABLE object_%s RENAME TO object_ids_table;' % G.nodes()[0])


        # 2.4 get the object_ids_table
        object_ids_df = pd.read_sql_query("SELECT *, 1 as cross_join_column FROM object_ids_table", connection)
        print('len(object_ids_df): ' + str(len(object_ids_df)))







    # ==================================================================================================
    # PART 3: get the remaining data
    # ==================================================================================================  
        print('part3')
        for object_number in object_numbers: 
            query_string = "SELECT DISTINCT obj%sattrobject_id FROM object_ids_table" % (object_number)
            cursor.execute(query_string)
            object_ids = [entry[0] for entry in cursor.fetchall()]
            print('len(obj%sattrobject_id): %s' % (object_number, len(object_ids)))


            sql_string5 = '''
                        SELECT 
                            'obj%sattr' || CAST(inner_query.attribute_id AS TEXT) AS column_name,
                            inner_query.object_id, 
                            inner_query.value_as_string,
                            inner_query.valid_time_start
                        FROM (
                                SELECT  object_id, 
                                        attribute_id, 
                                        value_as_string, 
                                        valid_time_start, 
                                        ROW_NUMBER() OVER(PARTITION BY object_id, attribute_id ORDER BY valid_time_start ASC, data_quality DESC) AS rank
                                FROM collection_data_point
                                WHERE object_id IN (
                                                      SELECT DISTINCT obj%sattrobject_id 
                                                      FROM object_ids_table
                                                    )
                                  AND valid_time_start >= %s
                                  AND valid_time_end <= %s 
                            )  as inner_query
                        WHERE inner_query.rank = 1
                    ''' % (object_number, object_number, valid_time_start, valid_time_end)
            long_table_df = pd.read_sql_query(sql_string5, connection)
            print('len(long_table_df for obj%s): %s' % (object_number, len(long_table_df)))
            data_querying_info['debug_info'] = {'valid_time_start':valid_time_start, 'valid_time_end':valid_time_end, 'number_of_object_ids_searched_for_in_query_5': len(object_ids),   'number_of_object_ids_found_in_query_5': len(long_table_df['object_id'].unique()), 'object_ids_searched_for_in_query_5': json.dumps(list([int(object_id) for object_id in object_ids])), 'object_ids_found_in_query_5': json.dumps([int(object_id) for object_id in list(long_table_df['object_id'].unique())])}


            # get data_querying_info['timestamps']
            data_timestamps_df = long_table_df.groupby('object_id').aggregate({'object_id':'first','valid_time_start':'min'})
            data_timestamps_df = data_timestamps_df.groupby('valid_time_start').aggregate({'valid_time_start':'first','object_id':pd.Series.nunique})
            data_timestamps_df.reset_index(inplace=True, drop=True)
            data_timestamps_df = data_timestamps_df.sort_values('valid_time_start')
            data_timestamps_df.index = data_timestamps_df['valid_time_start']
            data_timestamps_dict = data_timestamps_df['object_id'].to_dict()
            data_querying_info['timestamps'][object_number] = data_timestamps_dict


            # get broad_table_df
            long_table_df = long_table_df[['object_id', 'column_name', 'value_as_string']]
            long_table_df.set_index(['object_id','column_name'],inplace=True)
            broad_table_df = long_table_df.unstack(level=['column_name'])
            broad_table_df.columns = [col[1] for col in list(broad_table_df.columns)]

            # insert missing columns
            object_type_id = objects_dict[object_number]['object_type_id']
            list_of_parent_object_types = [el['id'] for el in get_from_db.get_list_of_parent_objects(object_type_id)]
            all_attribute_ids = Attribute.objects.filter(first_applicable_object_type__in = list_of_parent_object_types).values_list('id', flat=True)
            all_columns = ['obj%sattr%s' % (object_number, attribute_id) for attribute_id in all_attribute_ids]
            print('insert missing columns --------')
            print('list_of_parent_object_types: ' + str(list_of_parent_object_types))
            print('all_attribute_ids: ' + str(all_attribute_ids))
            existing_columns = list(broad_table_df.columns)
            for column_name in all_columns:
                if column_name not in existing_columns:
                    broad_table_df[column_name] = None


            object_ids_df = pd.merge(object_ids_df, broad_table_df, left_on='obj%sattrobject_id' % object_number, right_on='object_id', how='inner')
            print('len(object_ids_df): ' + str(len(object_ids_df)))


    # ==================================================================================================
    # PART 4: convert to correct datatype
    # ================================================================================================== 
    print('part4')
    attribute_data_types_dict = {attribute.id: attribute.data_type for attribute in list(Attribute.objects.all())}
    for column_name in object_ids_df.columns:
        if 'attr' in column_name and 'object_id' not in column_name:
            col_attribute_id = int(column_name.split('attr')[1])
            col_data_type = attribute_data_types_dict[col_attribute_id]

            if col_data_type in ['relation','int','real']:
                object_ids_df[column_name] = object_ids_df[column_name].astype(float)
            elif col_data_type in ['boolean','bool']:
                object_ids_df[column_name] = object_ids_df[column_name].replace({'True':True,'False':False})


    # save data_querying_info
    if (simulation_id is not None):
        simulation_model = Simulation_model.objects.get(id=simulation_id)
        simulation_model.data_querying_info = json.dumps(data_querying_info)
        simulation_model.save()


    return object_ids_df







def get_data_from_related_objects__multiple_timesteps(objects_dict, valid_time_start, valid_time_end, timestep_size, progress_tracking_file_name, max_number_of_instances, y0_columns=None):
    print('~~~~~~~~~~~~~~~~~  get_data_from_related_objects__multiple_timesteps  ~~~~~~~~~~~~~~~~~')
    print('valid_time_start:' + str(valid_time_start))
    print('valid_time_end:' + str(valid_time_end))
    print('timestep_size:' + str(timestep_size))
    sqlite_database = 'IS_USING_SQLITE_DB' in dict(os.environ).keys()
    object_numbers = list(objects_dict.keys())
    condition_holding_period_start = valid_time_start
    condition_holding_period_end = max((valid_time_end-valid_time_start)/3, valid_time_start + timestep_size) # valid_time_end equals either the half-time or the first timestep (whichever is later)

    # ==================================================================================================
    # PART 1: for every object get a table with the object id that fulfill the internal criteria
    # ==================================================================================================
    print('part1')
    with connection.cursor() as cursor:
        for object_number in object_numbers:

            object_type_id = objects_dict[object_number]['object_type_id']
            relation_ids = [relation['attribute_id'] for relation in objects_dict[object_number]['object_relations']]
            filter_facts  = objects_dict[object_number]['object_filter_facts']

            child_object_types = get_from_db.get_list_of_child_objects(object_type_id)
            child_object_type_ids = [el['id'] for el in child_object_types]
            child_object_types_string = ", ".join("'{0}'".format(object_type_id) for object_type_id in child_object_type_ids)


            # 1.1 create table with object_ids and potentially relation_ids
            cursor.execute('DROP TABLE IF EXISTS object_%s' % str(object_number))

            if len(relation_ids) == 0:                
                sql_string1 = """CREATE TEMPORARY TABLE object_%s AS 
                                    SELECT inner_query.* 
                                    FROM ( 
                                        SELECT DISTINCT object_id AS obj%sattrobject_id 
                                        FROM collection_data_point 
                                        WHERE valid_time_start <= %s 
                                          AND valid_time_end >= %s 
                                          AND object_id IN ( 
                                                            SELECT DISTINCT id  
                                                            FROM collection_object  
                                                            WHERE object_type_id IN (%s) 
                                                            )
                                """ % (str(object_number), str(object_number), str(condition_holding_period_end), str(condition_holding_period_start), child_object_types_string)

            elif len(relation_ids) == 1:
                                sql_string1 = """CREATE TEMPORARY TABLE object_%s AS 
                                    SELECT inner_query.* 
                                    FROM ( 
                                        SELECT related_objects.object_id AS obj%sattrobject_id, related_objects.numeric_value AS object_%s_relation_%s 
                                        FROM ( 
                                            SELECT DISTINCT object_id, numeric_value  
                                            FROM collection_data_point 
                                            WHERE attribute_id = '%s' 
                                              AND valid_time_start <= %s 
                                              AND valid_time_end >= %s 
                                              AND object_id IN ( 
                                                                SELECT DISTINCT id 
                                                                FROM collection_object 
                                                                WHERE object_type_id IN (%s) 
                                                                )
                                            ) related_objects
                                """ % (str(object_number), str(object_number), str(object_number), str(relation_ids[0]), str(relation_ids[0]), str(condition_holding_period_end), str(condition_holding_period_start), child_object_types_string)

            else: 
                cursor.execute('DROP TABLE IF EXISTS object_%s__with_missing_relations' % str(object_number))
                sql_string1 = """CREATE TEMPORARY TABLE object_%s__with_missing_relations AS 
                                    SELECT inner_query.* 
                                    FROM ( 
                                        SELECT related_objects.object_id, related_objects.numeric_value AS object_%s_relation_%s 
                                        FROM ( 
                                            SELECT DISTINCT object_id, numeric_value  
                                            FROM collection_data_point 
                                            WHERE attribute_id = '%s' 
                                              AND valid_time_start <= %s 
                                              AND valid_time_end >= %s 
                                              AND object_id IN ( 
                                                                SELECT DISTINCT id 
                                                                FROM collection_object 
                                                                WHERE object_type_id IN (%s) 
                                                                )
                                            ) related_objects 
                                """ % (str(object_number), str(object_number), str(relation_ids[0]), str(relation_ids[0]), str(condition_holding_period_end), str(condition_holding_period_start), child_object_types_string)


            if len(filter_facts)>0:
                if len(relation_ids) > 0:
                    sql_string1    += ''' INNER JOIN ( '''
                else: 
                    sql_string1    += ''' INTERSECT '''



                for fact_index, filter_fact in enumerate(filter_facts):
                    if fact_index > 0:
                        sql_string1 += ''' INTERSECT '''

                    sql_string1    += '''   SELECT DISTINCT object_id 
                                            FROM collection_data_point 
                                            WHERE 
                    '''
                    if filter_fact['operation'] == '=':     
                        sql_string1 +=            "attribute_id = '%s' AND string_value = '%s' AND valid_time_start <= %s AND valid_time_end >= %s " % (filter_fact['attribute_id'], filter_fact['value'], str(condition_holding_period_end), str(condition_holding_period_start))
                    elif filter_fact['operation'] == '>':
                        sql_string1 +=            "attribute_id = '%s' AND numeric_value > %s AND valid_time_start <= %s AND valid_time_end >= %s " % (filter_fact['attribute_id'], filter_fact['value'], str(condition_holding_period_end), str(condition_holding_period_start))
                    elif filter_fact['operation'] == '<':
                        sql_string1 +=            "attribute_id = '%s' AND numeric_value < %s AND valid_time_start <= %s AND valid_time_end >= %s " % (filter_fact['attribute_id'], filter_fact['value'], str(condition_holding_period_end), str(condition_holding_period_start))
                    elif filter_fact['operation'] == 'in':
                        values = ['"%s"' % value for value in filter_fact['value']]
                        sql_string1 +=            "attribute_id = '%s' AND string_value IN (%s) AND valid_time_start <= %s AND valid_time_end >= %s " % (filter_fact['attribute_id'], ', '.join(values), str(condition_holding_period_end), str(condition_holding_period_start))

                
                if y0_columns is not None:
                    attribute_ids = [col.split('attr')[1] for col in y0_columns if 'obj' + str(object_number) + 'attr' in col]
                    for attribute_id in attribute_ids:
                        sql_string1    += '''INTERSECT 
                                                SELECT DISTINCT object_id 
                                                FROM collection_data_point 
                                                WHERE attribute_id = '%s' 
                                                  AND valid_time_start <= %s 
                                                  AND valid_time_end >= %s 
                                            INTERSECT 
                                                SELECT DISTINCT object_id 
                                                FROM collection_data_point 
                                                WHERE attribute_id = '%s' 
                                                  AND valid_time_start <= %s 
                                                  AND valid_time_end >= %s 
                    ''' % (attribute_id, str(condition_holding_period_end), str(condition_holding_period_start), attribute_id, str(valid_time_end), str(condition_holding_period_end))
                    
                if len(relation_ids) > 0:
                    sql_string1    += ''') fact_objects 
                                        ON fact_objects.object_id = related_objects.object_id ''' 

                
            print('1.1')    
            sql_string1    += '''  ) as inner_query 
                                ORDER BY RANDOM() 
                                LIMIT %s;''' % max_number_of_instances
            cursor.execute(sql_string1)




            # 1.2 add any missing relation_ids to the table
            print('1.2')
            if len(relation_ids) > 1: 
                missing_relations = relation_ids[1:]
                missing_relations_string = ", ".join("relation_table_%s.object_%s_relation_%s" % (relation_id, object_number, relation_id) for relation_id in missing_relations)

                sql_string2 = """CREATE TEMPORARY TABLE object_%s AS 
                                    SELECT original_table.object_id AS obj%sattrobject_id, original_table.object_%s_relation_%s, %s 
                                    FROM object_%s__with_missing_relations AS original_table 

                            """ % (str(object_number), str(object_number), str(object_number), str(relation_ids[0]), missing_relations_string, str(object_number))

                for missing_relation in missing_relations:
                    sql_string2 += """INNER JOIN ( 
                                            SELECT DISTINCT object_id, numeric_value AS object_%s_relation_%s 
                                            FROM collection_data_point 
                                            WHERE attribute_id = '%s' 
                                              AND valid_time_start >= %s 
                                              AND valid_time_end <= %s 
                                              AND object_id IN (SELECT DISTINCT object_id FROM object_%s__with_missing_relations) 
                                    ) AS relation_table_%s 
                                    ON original_table.object_id = relation_table_%s.object_id  
                                """ % (str(object_number), str(missing_relation), str(missing_relation), str(condition_holding_period_start), str(condition_holding_period_end), str(object_number), str(missing_relation), str(missing_relation))
                cursor.execute(sql_string2)
            print('1.3')


    # ==================================================================================================
    # PART 2: join the object-tables
    # ==================================================================================================   
        with open(progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 3, "total_number": 6}))

        print('part2')
        # 2.1 prepare directed graph 
        G = nx.MultiDiGraph()
        node_sizes = {}
        edge_counter = 0
        edge_info = {}

        for object_number in object_numbers: 
            # populate node_sizes    
            # if sqlite_database:
            cursor.execute("SELECT COUNT(DISTINCT obj%sattrobject_id) AS approximate_row_count FROM object_%s;" % (object_number, object_number))
            # else:
            #     cursor.execute("SELECT reltuples AS approximate_row_count FROM pg_class WHERE relname = 'object_%s';")
            objects_table_length = 0
            objects_table_length = cursor.fetchall()[0][0]
            node_sizes[object_number] = objects_table_length
            print('len(table' + str(object_number) + ') = ' + str(objects_table_length))

            # add nodes and edges (the edge-weight is an id used for storing additional edge_info)
            G.add_node(object_number)
            for relation in objects_dict[object_number]['object_relations']:
                edge_counter += 1
                G.add_edge(object_number, str(relation['target_object_number']), weight=edge_counter)
                edge_info[edge_counter] = {'related_object_nb': str(relation['target_object_number']), 'relation_column_name': 'object_%s_relation_%s' % (object_number, relation['attribute_id'])}



        # 2.2 sequential collapse
        while len(G.edges()) > 0:
            # sort the edges - we always collapse/contract the edge with the lowest score
            edges_with_scores = [(edge[3], (edge[0], edge[1]), (node_sizes[str(edge[0])]*node_sizes[str(edge[1])]), edge[2]) for edge in list(G.edges_iter(data='weight', keys=True))]
            # pattern: [(<weight=edge_number>, (<origin_object_nb>, <target_object_nb>), <edge_score>, <edge_key>), ...]
            edges_with_scores = sorted(edges_with_scores, key=lambda tup: tup[2])
            edge_number = edges_with_scores[0][0] 
            origin_object_nb = edges_with_scores[0][1][0]
            target_object_nb = edges_with_scores[0][1][1] 
            edge_key = edges_with_scores[0][3]
            relation_column_name = edge_info[edge_number]['relation_column_name']
            related_object_nb = edge_info[edge_number]['related_object_nb']


            # the target object will be fused into the origin i.e. the target is removed
            # collapsing the tables
            sql_string3_1 = '''ALTER TABLE object_%s RENAME TO object_%s__temp;''' % (origin_object_nb, origin_object_nb)
            cursor.execute(sql_string3_1)
            
            if origin_object_nb == target_object_nb:
                target_table_name = 'object_%s__temp' % target_object_nb
            else:
                target_table_name = 'object_%s' % target_object_nb    

            sql_string3_2 = ''' CREATE TEMPORARY TABLE object_%s AS 
                                SELECT origin.*, target.* 
                                FROM object_%s__temp AS origin 
                                INNER JOIN ( 
                                    SELECT * FROM %s  
                                ) AS target 
                                ON origin.%s = target.obj%sattrobject_id  
                                WHERE origin.obj%sattrobject_id <> target.obj%sattrobject_id  
                                ORDER BY random()
                                LIMIT %s;
                        ''' % (origin_object_nb, origin_object_nb, target_table_name, relation_column_name, related_object_nb, origin_object_nb, related_object_nb, max_number_of_instances)
            cursor.execute(sql_string3_2)
            sql_string3_3 = '''DROP TABLE object_%s__temp;''' % origin_object_nb
            cursor.execute(sql_string3_3)

            # collapsing the graph
            G.remove_edge(str(origin_object_nb), str(target_object_nb),key=edge_key)
            G = nx.contracted_nodes(G, str(origin_object_nb), str(target_object_nb))

            # updating node_sizes
            cursor.execute("SELECT COUNT(*) FROM object_%s;" % origin_object_nb)
            new_origin_table_length = cursor.fetchall()[0][0]
            node_sizes[str(origin_object_nb)] = new_origin_table_length
            del node_sizes[str(target_object_nb)] 
            print('collapsing - table length = ' + str(new_origin_table_length))



        # 2.3 cross-join the remaining nodes/tables
        cursor.execute('''DROP TABLE IF EXISTS object_ids_table;''')
        if len(G.nodes()) > 1:

            select_columns = ", ".join("object_%s.*" % (node_id) for node_id in list(G.nodes()))
            sql_string4 = '''  
                        CREATE TEMPORARY TABLE object_ids_table AS 
                            SELECT %s 
                            FROM object_%s  
                        ''' % (select_columns, list(G.nodes())[0])
            for node_id in list(G.nodes())[1:]:
                sql_string4 += 'CROSS JOIN object_%s ' % node_id
            sql_string4 += ';' 
            cursor.execute(sql_string4)
        else:
            cursor.execute('ALTER TABLE object_%s RENAME TO object_ids_table;' % G.nodes()[0])

        
        # 2.4 get the object_ids_table
        object_ids_df = pd.read_sql_query("SELECT *, 1 as cross_join_column FROM object_ids_table", connection)


        # TESTING ----------------------------------------------------
        # object_ids_df.to_csv('C:/Users/l412/Documents/2 temporary stuff/2020-06-11/object_ids_df.csv', index=False)
        # ------------------------------------------------------------

        # number_of_periods = int(np.ceil(valid_time_end - valid_time_start)/timestep_size)
        # periods_df = pd.DataFrame({'period': range(number_of_periods), 'cross_join_column': [1]*number_of_periods})
        # object_ids_df = pd.merge(object_ids_df, periods_df, on='cross_join_column')

        

    # ==================================================================================================
    # PART 3: get the remaining data
    # ==================================================================================================  
        with open(progress_tracking_file_name, "w") as progress_tracking_file:
            progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 4, "total_number": 6}))

        print('part3')
        for object_number in object_numbers: 

            query_string = "SELECT DISTINCT obj%sattrobject_id FROM object_ids_table" % (object_number)
            cursor.execute(query_string)
            object_ids = [entry[0] for entry in cursor.fetchall()]
            if sqlite_database:
                sql_string5 = '''
                            SELECT 
                                'obj%sattr' || CAST(attribute_id AS TEXT) || 'period' || CAST(period AS TEXT) AS column_name, 
                                object_id, 
                                --attribute_id, 
                                value_as_string, 
                                period 
                            FROM ( 
                                    SELECT  object_id, 
                                            attribute_id, 
                                            value_as_string, 
                                            valid_time_start, 
                                            CAST((valid_time_start-(%s))/%s AS INT) AS period, 
                                            ROW_NUMBER() OVER(PARTITION BY object_id, attribute_id, CAST((valid_time_start-(%s))/%s AS INT) ORDER BY data_quality DESC,upload_id DESC) AS rank 
                                    FROM collection_data_point 
                                    WHERE object_id IN ( 
                                                          SELECT DISTINCT obj%sattrobject_id 
                                                          FROM object_ids_table 
                                                        ) 
                                      AND valid_time_start >= %s 
                                      AND valid_time_end <= %s  
                                )
                            WHERE rank = 1
                        ''' % (object_number, valid_time_start, timestep_size, valid_time_start, timestep_size, object_number, valid_time_start, valid_time_end)
            else:
                sql_string5 = '''
                            SELECT 
                                inner_query.object_id, 
                                concat('obj%sattr', inner_query.attribute_id, 'period', inner_query.period) AS column_name, 
                                --inner_query.attribute_id, 
                                inner_query.value_as_string, 
                                inner_query.period 
                            FROM ( 
                                    SELECT  object_id, 
                                            attribute_id, 
                                            value_as_string, 
                                            valid_time_start, 
                                            FLOOR((valid_time_start-(%s))/%s) AS period, 
                                            ROW_NUMBER() OVER(PARTITION BY object_id, attribute_id, FLOOR((valid_time_start-(%s))/%s) ORDER BY data_quality DESC,upload_id DESC) AS rank 
                                    FROM collection_data_point 
                                    WHERE object_id IN ( 
                                                          SELECT DISTINCT obj%sattrobject_id 
                                                          FROM object_ids_table 
                                                        ) 
                                      AND valid_time_start >= %s 
                                      AND valid_time_end <= %s 
                                ) as inner_query 
                            WHERE inner_query.rank = 1
                        ''' % (object_number, valid_time_start, timestep_size, valid_time_start, timestep_size, object_number, valid_time_start, valid_time_end)

            long_table_df = pd.read_sql_query(sql_string5, connection)
            
            # TESTING ----------------------------------------------------
            if object_number == 17 or object_number == '17':
                print('[[[[[[[[[[[[[[[[[[  logging obj17  ]]]]]]]]]]]]]]]]]]]]]]')
                generally_useful_functions.log(sql_string5, 'sql_string5')
                generally_useful_functions.log(long_table_df, 'long_table_df__object_' + str(object_number))
                distinct_object_ids_df = pd.read_sql_query("SELECT DISTINCT obj1attrobject_id FROM object_ids_table", connection)
                generally_useful_functions.log(distinct_object_ids_df, 'distinct_object_ids_df')
                generally_useful_functions.log(object_ids_df, 'object_ids_df')
            # ------------------------------------------------------------
            
            long_table_df.set_index(['object_id','column_name','period'],inplace=True)
            broad_table_df = long_table_df.unstack(level=['column_name', 'period'])
            # long_table_df.set_index(['object_id','column_name','period', 'attribute_id'],inplace=True)
            # long_table_df.unstack(level=['column_name', 'period', 'attribute_id'])
            broad_table_df.columns = [col[1] for col in list(broad_table_df.columns)]
            
            # TESTING ----------------------------------------------------
            if object_number == 17 or object_number == '17':
                print('[[[[[[[[[[[[[[[[[[  logging obj17  ]]]]]]]]]]]]]]]]]]]]]]')
                generally_useful_functions.log(broad_table_df, 'broad_table_df')
            # ------------------------------------------------------------

            object_ids_df = pd.merge(object_ids_df, broad_table_df, left_on='obj%sattrobject_id' % object_number, right_on='object_id', how='inner')
            print('joining remaining data obj %s (%s rows) - table length = %s' %  (object_number, len(broad_table_df), len(object_ids_df)))

    
    # ==================================================================================================
    # PART 4: convert to correct datatype
    # ================================================================================================== 
    with open(progress_tracking_file_name, "w") as progress_tracking_file:
        progress_tracking_file.write(json.dumps({"text": 'Initializing simulations - step: ', "current_number": 5, "total_number": 6}))
    print('part4')
    attribute_data_types_dict = {attribute.id: attribute.data_type for attribute in list(Attribute.objects.all())}
    for column_name in object_ids_df.columns:
        if 'period' in column_name:
            col_attribute_id = int(column_name.split('attr')[1].split('period')[0])
            col_data_type = attribute_data_types_dict[col_attribute_id]

            if col_data_type in ['relation','int','real']:
                object_ids_df[column_name] = object_ids_df[column_name].astype(float)
            elif col_data_type in ['boolean','bool']:
                object_ids_df[column_name] = object_ids_df[column_name].replace({'True':True,'False':False})

    print('converted datatypes - table length = ' + str(len(object_ids_df)))
    return object_ids_df










# used by simulation.py
@generally_useful_functions.cash_result
def get_data_from_related_objects__multiple_timesteps__old(objects_dict, times, timestep_size):
    print('~~~~~~~~~~~~~~~  get_data_from_related_objects__multiple_timesteps  ~~~~~~~~~~~~~~~~~~~~~~')
    object_numbers = list(objects_dict.keys())

    
    object_and_period_tables = {}
    object_data_tables = {}
    for object_number in object_numbers:

        # PART1: create object_data_tables - i.e. get broad_table_df for every object_number and period
        object_and_period_tables[object_number] = {}

        # get object_ids
        print('object_number: ' + str(object_number))
        object_type_id = objects_dict[object_number]['object_type_id']
        filter_facts  = objects_dict[object_number]['object_filter_facts']
        child_object_types = get_from_db.get_list_of_child_objects(object_type_id)
        child_object_ids = [el['id'] for el in child_object_types]

        with connection.cursor() as cursor:
            query_string = "SELECT DISTINCT id FROM collection_object WHERE object_type_id IN (%s);" % (", ".join("'{0}'".format(object_type_id) for object_type_id in child_object_ids))
            cursor.execute(query_string)
            print(query_string)
            object_ids = [entry[0] for entry in cursor.fetchall()]


        for period in range(len(times)):

            # get broad_table_df
            period_start_time = times[period]
            period_end_time = times[period] + timestep_size
            broad_table_df = filter_and_make_df_from_datapoints(object_type_id, object_ids, filter_facts, period_start_time, period_end_time)  
            print('================================    filter_and_make_df_from_datapoints    ==========================================')
            print('object_type_id: ' + object_type_id)
            print('filter_facts: ' + str(filter_facts))
            print('period_start_time: ' + str(period_start_time))
            print('period_end_time: ' + str(period_end_time))
            print('broad_table_df is None: ' + str(broad_table_df is None))
            if broad_table_df is not None:
                print('len(broad_table_df): ' + str(len(broad_table_df)))


            # if broad_table_df is None or len(broad_table_df) == 0:
            #     return pd.DataFrame({'obj' + str(object_number) + 'attrobject_id':[] for object_number in object_numbers})

            print('3')
            if broad_table_df is not None:
                # broad_table_df = broad_table_df[[column for column in broad_table_df.columns if column in y0_columns + ['object_id'] ]]
                broad_table_df.columns = ['obj' + str(object_number) + 'attrobject_id' if column == 'object_id' else 'obj' + str(object_number) + 'attr' + str(column) + 'period' + str(period)  for column in broad_table_df.columns]
                broad_table_df['cross_join_column'] = 1
                object_and_period_tables[object_number][period] = broad_table_df
            else:
                object_and_period_tables[object_number][period] = pd.DataFrame(columns=['cross_join_column','obj' + str(object_number) + 'attrobject_id'])


    
        # PART2: for each object merge the periods
        object_data_tables[object_number] = pd.DataFrame(columns=['cross_join_column','obj' + str(object_number) + 'attrobject_id'])
        for period in range(len(times)):
            object_data_tables[object_number] = pd.merge(object_data_tables[object_number], object_and_period_tables[object_number][period], on=['cross_join_column','obj' + str(object_number) + 'attrobject_id'],  how='outer')

        if len(object_data_tables[object_number]) == 0:
            object_data_tables[object_number] = pd.DataFrame({'cross_join_column': [1]*len(object_ids), 'obj' + str(object_number) + 'attrobject_id': object_ids})
      


    # PART3: merge the object_data_tables according to the relations
    print('PART3: merge the object_data_tables according to the relations')
    merged_object_data_tables = pd.DataFrame({'cross_join_column':[1]})
    list_of_added_tables = []

    for object_number in object_numbers:
        print('4')
        print('object_number: ' + str(object_number))
        print('len(object_data_tables[object_number]): ' + str(len(object_data_tables[object_number])))
        if object_number not in list_of_added_tables:
            merged_object_data_tables = pd.merge(merged_object_data_tables , object_data_tables[object_number] , on='cross_join_column', how='inner')
            print('4.5')
            print('len(merged_object_data_tables): ' + str(len(merged_object_data_tables)))
            merged_object_data_tables = make_the_columns_for_joining_relations_on(merged_object_data_tables, object_number, objects_dict, times)
            list_of_added_tables.append(object_number)
    

        print('5')
        object_relations = objects_dict[object_number]['object_relations']
        for relation in object_relations:
            print('6.1 - ' + str(relation['attribute_id'])  )
            target_object_number = str(relation['target_object_number'])
            attribute_id_column = 'obj' + str(object_number) + 'attr' + str(relation['attribute_id']) 
            print('6.2')

            merged_object_data_tables[attribute_id_column] = pd.to_numeric(merged_object_data_tables[attribute_id_column])
            merged_object_data_tables = pd.merge(merged_object_data_tables, object_data_tables[str(target_object_number)], left_on=attribute_id_column, right_on='obj' +str(target_object_number) + 'attrobject_id', how='inner', suffixes=('-old', ''))
            merged_object_data_tables = merged_object_data_tables.drop_duplicates(subset=['obj' + object_nb + 'attrobject_id' for object_nb in list_of_added_tables])
            print('len(merged_object_data_tables): ' + str(len(merged_object_data_tables)))
            print('6.3')
            if target_object_number not in list_of_added_tables:
                merged_object_data_tables = make_the_columns_for_joining_relations_on(merged_object_data_tables, object_number, objects_dict, times)
                list_of_added_tables.append(target_object_number)
                print('6.4')
            else:
                merged_object_data_tables[merged_object_data_tables['obj' + target_object_number + 'attrobject_id']==merged_object_data_tables['obj' + str(target_object_number) + 'attrobject_id-old']]
                columns_without_old = [col for col in merged_object_data_tables.columns if col[-4:]!='-old']
                merged_object_data_tables = merged_object_data_tables[columns_without_old]
                print('6.5')

             # merged_object_data_tables = merged_object_data_tables[[column for column in merged_object_data_tables.columns if len(column.split('attr')) <3]]
    return merged_object_data_tables




def make_the_columns_for_joining_relations_on(merged_object_data_tables, object_number, objects_dict, times):
    for relation in objects_dict[object_number]['object_relations']:
        new_column = 'obj' + str(object_number) + 'attr' + str(relation['attribute_id']) 
        merged_object_data_tables[new_column] = None
        columns_to_combine = [new_column + 'period' + str(period) for period in range(len(times))]
        columns_to_combine = [column for column in columns_to_combine if column in merged_object_data_tables.columns]
        for column_to_combine in columns_to_combine:
            merged_object_data_tables[new_column] = merged_object_data_tables[new_column].combine_first(merged_object_data_tables[column_to_combine])
    return merged_object_data_tables






# used by simulation.py
def get_data_from_related_objects(objects_dict, specified_start_time, specified_end_time, simulation_id=None):
    print('~~~~~~~~~~~~~~~  get_data_from_related_objects  ~~~~~~~~~~~~~~~~~~~~~~')
    object_numbers = list(objects_dict.keys())

    # PART1: create object_data_tables - i.e. get broad_table_df for every object_number
    object_data_tables = {}
    data_querying_info__table_sizes = {}
    data_querying_info__relation_sizes = {}
    for object_number in object_numbers:
        print('object_number: ' + str(object_number))
        # get object_ids
        object_type_id = objects_dict[object_number]['object_type_id']
        filter_facts  = objects_dict[object_number]['object_filter_facts']
        child_object_types = get_from_db.get_list_of_child_objects(object_type_id)
        child_object_ids = [el['id'] for el in child_object_types]
        print('1')

        with connection.cursor() as cursor:
            query_string = "SELECT DISTINCT id FROM collection_object WHERE object_type_id IN (%s);" % (", ".join("'{0}'".format(object_type_id) for object_type_id in child_object_ids))
            cursor.execute(query_string)
            print(query_string)
            object_ids = [entry[0] for entry in cursor.fetchall()]

         # broad_table_df and data_querying_info
        print('2')
        if simulation_id is not None:
            saving_details_for_data_querying_info = {'simulation_id':simulation_id, 'object_number':object_number}
            broad_table_df = filter_and_make_df_from_datapoints(object_type_id, object_ids, filter_facts, specified_start_time, specified_end_time, saving_details_for_data_querying_info)  
        else: 
            broad_table_df = filter_and_make_df_from_datapoints(object_type_id, object_ids, filter_facts, specified_start_time, specified_end_time)  

        if broad_table_df is None or len(broad_table_df) == 0:
            return pd.DataFrame({'obj' + str(object_number) + 'attrobject_id':[] for object_number in object_numbers})

        print('3')
        if broad_table_df is not None:
            broad_table_df.columns = ['obj' + str(object_number) + 'attr' + str(column) for column in broad_table_df.columns]
            broad_table_df['cross_join_column'] = 1
            object_data_tables[object_number] = broad_table_df
        else:
            object_data_tables[object_number] = pd.DataFrame(columns=['cross_join_column','obj' + str(object_number) + 'attrobject_id'])




    # PART2: merge the broad_table_dfs according to the relations
    merged_object_data_tables = pd.DataFrame({'cross_join_column':[1]})
    list_of_added_tables = []

    for object_number in object_numbers:
        print('object_number: ' + str(object_number))
        data_querying_info__table_sizes[object_number] = {'number_of_objects':len(object_data_tables[object_number]),'number_of_matches':{}}
        if object_number not in list_of_added_tables:
            print('len(merged_object_data_tables) before: ' + str(len(merged_object_data_tables)))
            print(str(object_number))
            print(str(len(object_data_tables[str(object_number)])))
            merged_object_data_tables = pd.merge(merged_object_data_tables , object_data_tables[object_number] , on='cross_join_column', how='inner')
            print('len(merged_object_data_tables) after: ' + str(len(merged_object_data_tables)))
            list_of_added_tables.append(object_number)
            


        object_relations = objects_dict[object_number]['object_relations']
        for relation in object_relations:
            target_object_number = str(relation['target_object_number'])
            attribute_id_column = 'obj' + str(object_number) + 'attr' + str(relation['attribute_id']) 

            # data_querying_info
            source = list(object_data_tables[str(object_number)]['obj' + str(object_number) + 'attr' + str(relation['attribute_id'])].astype('int'))
            target = list(object_data_tables[str(target_object_number)]['obj' +str(target_object_number) + 'attrobject_id'])
            matched_from_source = len([el for el in source if el in set(target)])
            matched_from_target = len([el for el in target if el in set(source)])
            link_id = str(object_number) + ',' + str(relation['attribute_id']) + ',' + str(target_object_number)
            data_querying_info__relation_sizes[link_id] = { 'matched_from_source':matched_from_source,
                                                            'matched_from_target':matched_from_target}
            # merge
            merged_object_data_tables[attribute_id_column] = pd.to_numeric(merged_object_data_tables[attribute_id_column])
            print('len(merged_object_data_tables) before2: ' + str(len(merged_object_data_tables)))
            print(str(target_object_number))
            print(str(len(object_data_tables[str(target_object_number)])))
            merged_object_data_tables = pd.merge(merged_object_data_tables, object_data_tables[str(target_object_number)], left_on=attribute_id_column, right_on='obj' +str(target_object_number) + 'attrobject_id', how='inner', suffixes=('-old', ''))
            merged_object_data_tables = merged_object_data_tables.drop_duplicates(subset=['obj' + object_nb + 'attrobject_id' for object_nb in list_of_added_tables])
            print('len(merged_object_data_tables) after2: ' + str(len(merged_object_data_tables)))
            if target_object_number not in list_of_added_tables:
                list_of_added_tables.append(target_object_number)

            else:
                merged_object_data_tables[merged_object_data_tables['obj' + target_object_number + 'attrobject_id']==merged_object_data_tables['obj' + str(target_object_number) + 'attrobject_id-old']]
                columns_without_old = [col for col in merged_object_data_tables.columns if col[-4:]!='-old']
                merged_object_data_tables = merged_object_data_tables[columns_without_old]


    # data_querying_info
    print('data querying info')
    if (simulation_id is not None):
        simulation_model = Simulation_model.objects.get(id=saving_details_for_data_querying_info['simulation_id'])
        data_querying_info = json.loads(simulation_model.data_querying_info)
        data_querying_info['table_sizes'] = data_querying_info__table_sizes
        data_querying_info['relation_sizes'] = data_querying_info__relation_sizes
        simulation_model.data_querying_info = json.dumps(data_querying_info)
        simulation_model.save()

    return merged_object_data_tables





# used in query_data.html
def get_data_from_objects_behind_the_relation(object_type_id, object_ids, specified_start_time, specified_end_time):


    filter_facts = []
    broad_table_df = filter_and_make_df_from_datapoints(object_type_id, object_ids, filter_facts, specified_start_time, specified_end_time)   

    # prepare response
    if broad_table_df is not None:

        # for response: list of the tables' attributes sorted with best populated first
        related_table_attributes = []
        sorted_attribute_ids = broad_table_df.isnull().sum(0).sort_values(ascending=False).index
        sorted_attribute_ids = [int(attribute_id) for attribute_id in list(sorted_attribute_ids) if attribute_id.isdigit()]
        for attribute_id in sorted_attribute_ids:
            attribute_record = Attribute.objects.get(id=attribute_id)
            related_table_attributes.append({'attribute_id':attribute_id, 'attribute_name':attribute_record.name, 'attribute_data_type':attribute_record.data_type})

        # sort broad_table_df (the best-populated entities to the top)
        broad_table_df = broad_table_df.loc[broad_table_df.isnull().sum(1).sort_values().index]
        broad_table_df.index = broad_table_df['object_id']
        # object_ids_df = pd.DataFrame({'object_id':object_ids})
        # broad_table_df = pd.merge(object_ids_df, broad_table_df, on='object_id', how='left')

        response = {}
        response['related_table_body'] = broad_table_df.to_dict('dict')
        response['related_table_attributes'] = related_table_attributes
    else: 
        response = {}
        response['related_table_body'] = {}
        response['related_table_attributes'] = []

    return response





# used in learn_rule.py (triggered by learn_rule.html)
def get_training_data(object_type_id, filter_facts, valid_times): 
    
    # basic filtering: object_type and specified time range
    child_object_types = get_from_db.get_list_of_child_objects(object_type_id)
    child_object_ids = [el['id'] for el in child_object_types]

    with connection.cursor() as cursor:
        query_string = 'SELECT DISTINCT id FROM collection_object WHERE object_type_id IN (%s);' % (', '.join("'{0}'".format(object_type_id) for object_type_id in child_object_ids))
        cursor.execute(query_string)
        object_ids = [entry[0] for entry in cursor.fetchall()]

    broad_table_df = pd.DataFrame()
    for valid_time in valid_times:
        broad_table_df = broad_table_df.append(filter_and_make_df_from_datapoints(object_type_id, object_ids, filter_facts, valid_time[0], valid_time[1]), sort=False)

    broad_table_df.columns = ['attr'+ col for col in broad_table_df.columns]
    broad_table_df.index = range(len(broad_table_df))
    broad_table_df.fillna(value=pd.np.nan, inplace=True)
    return broad_table_df




def filter_and_make_df_from_datapoints(object_type_id, object_ids, filter_facts, specified_start_time, specified_end_time, saving_details_for_data_querying_info=None):

    if len(object_ids) == 0:
        return None

    with connection.cursor() as cursor:

        cursor.execute('DROP TABLE IF EXISTS unfiltered_object_ids')
        sql_string1 = '''
            CREATE TEMPORARY TABLE unfiltered_object_ids AS
                SELECT DISTINCT object_id
                FROM collection_data_point
                WHERE valid_time_start >= %s
                  AND valid_time_start < %s
                  AND object_id IN (%s)
        ''' 
        object_ids = [str(object_id) for object_id in object_ids]
        cursor.execute(sql_string1 % (specified_start_time, specified_end_time, ','.join(object_ids)))


        # apply filter-facts
        print('2.1')
        query = cursor.execute('SELECT object_id FROM unfiltered_object_ids')
        unfiltered_object_ids = cursor.fetchall()
        if unfiltered_object_ids is None:
            return None

        else:

            valid_ranges_df = pd.DataFrame({'object_id':[result[0] for result in unfiltered_object_ids]})
            valid_ranges_df['valid_range'] = [[[specified_start_time,specified_end_time]] for i in valid_ranges_df.index]


            for fact_index, filter_fact in enumerate(filter_facts):
                # if 'DATABASE_URL' in dict(os.environ).keys() and dict(os.environ)['DATABASE_URL'][:8]!='postgres':
                if 'IS_USING_SQLITE_DB' in dict(os.environ).keys():
                    # SQLITE...
                    sql_string2 = '''
                            SELECT object_id, '[' || group_concat('[' || valid_time_start || ',' || valid_time_end || ']', ',') || ']' AS new_valid_range
                            FROM collection_data_point
                            WHERE object_id IN (SELECT object_id FROM unfiltered_object_ids)
                              AND 
                    '''
                else:
                    # POSTGES... 
                    sql_string2 = '''
                            SELECT object_id, '[' || string_agg('[' || valid_time_start || ',' || valid_time_end || ']', ',') || ']' AS new_valid_range
                            FROM collection_data_point
                            WHERE object_id IN (SELECT object_id FROM unfiltered_object_ids)
                              AND 
                    '''   
                

                if filter_fact['operation'] == '=':     
                    sql_string2 += "attribute_id = '%s' AND string_value = '%s'" % (filter_fact['attribute_id'], filter_fact['value'])
                elif filter_fact['operation'] == '>':
                    sql_string2 += "attribute_id = '%s' AND numeric_value > %s" % (filter_fact['attribute_id'], filter_fact['value'])
                elif filter_fact['operation'] == '<':
                    sql_string2 += "attribute_id = '%s' AND numeric_value < %s" % (filter_fact['attribute_id'], filter_fact['value'])
                elif filter_fact['operation'] == 'in':
                    values = ['"%s"' % value for value in filter_fact['value']]
                    sql_string2 += "attribute_id = '%s' AND string_value IN (%s)" % (filter_fact['attribute_id'], ', '.join(values))

                sql_string2 += " GROUP BY object_id "
                print('2.2')
                new_valid_ranges_df = pd.read_sql_query(sql_string2, connection)
                print('2.3')
                new_valid_ranges_df['new_valid_range'] = new_valid_ranges_df['new_valid_range'].apply(json.loads)
                new_valid_ranges_df['object_id'] = new_valid_ranges_df['object_id'].astype(int)
                
                # find the intersecting time ranges (= the overlap between the known valid_ranges and the valid_ranges from the new filter fact)
                print('2.4')
                valid_ranges_df = pd.merge(valid_ranges_df, new_valid_ranges_df, on='object_id', how='left')
                valid_ranges_df = valid_ranges_df[valid_ranges_df['new_valid_range'].notnull()]
                if len(valid_ranges_df) == 0:
                    return None
                valid_ranges_df['valid_range'] = valid_ranges_df.apply(generally_useful_functions.intersections, axis=1)
                valid_ranges_df = valid_ranges_df[['object_id', 'valid_range']]



            # choose the first time interval that satisfies all filter-fact conditions
            print('2.5')
            valid_ranges_df['satisfying_time_start'] = [object_ranges[0][0] if len(object_ranges)>0 else None for object_ranges in valid_ranges_df['valid_range'] ]
            valid_ranges_df['satisfying_time_end'] = [object_ranges[0][1] if len(object_ranges)>0 else None for object_ranges in valid_ranges_df['valid_range']]
            valid_ranges_df = valid_ranges_df[valid_ranges_df['satisfying_time_start'].notnull()]

            # make long table with all datapoints of the found objects
            print('2.6')
            unfiltered_object_ids = [str(result[0]) for result in unfiltered_object_ids]
            if len(unfiltered_object_ids) == 0:
                return None
            sql_string3 = 'SELECT object_id, attribute_id, value_as_string, numeric_value, string_value, boolean_value, valid_time_start, valid_time_end, data_quality FROM collection_data_point WHERE object_id IN (%s)' % (','.join(unfiltered_object_ids))
            print('2.6.1')
            print('2.6.1.1')
            long_table_df = pd.read_sql_query(sql_string3, connection)
            print('2.6.1.2')

            # found_objects = list(set(data_point_records.values_list('object_id', flat=True)))
            # all_data_points = Data_point.objects.filter(object_id__in=found_objects)    
            # long_table_df = pd.DataFrame(list(all_data_points.values()))


            # filter out the observations from not-satisfying times
            print('2.6.2')
            long_table_df = pd.merge(long_table_df, valid_ranges_df, how='inner', on='object_id')
            print('2.6.3')
            long_table_df = long_table_df[(long_table_df['valid_time_end'] > long_table_df['satisfying_time_start']) & (long_table_df['valid_time_start'] < long_table_df['satisfying_time_end'])]


            # select satisfying time (and remove the records from other times)
            print('2.8')
            total_data_quality_df = long_table_df.groupby(['object_id','satisfying_time_start']).aggregate({'object_id':'first','satisfying_time_start':'first', 'data_quality': np.sum, 'attribute_id': 'count'})
            total_data_quality_df = total_data_quality_df.rename(columns={"data_quality": "total_data_quality", "attribute_id":"attriubte_count"})

            print('2.9')
            total_data_quality_df.index = range(len(total_data_quality_df))
            total_data_quality_df = total_data_quality_df.sort_values(['total_data_quality','satisfying_time_start'], ascending=[False, True])
            total_data_quality_df = total_data_quality_df.drop_duplicates(subset=['object_id'], keep='first')
            long_table_df = pd.merge(long_table_df, total_data_quality_df, how='inner', on=['object_id','satisfying_time_start'])


            # remove the duplicates (=duplicate values within the satisfying time)
            print('2.10 - ' + str(len(long_table_df)))
            long_table_df['time_difference_of_start'] = abs(long_table_df['satisfying_time_start'] - long_table_df['valid_time_start'])
            long_table_df = long_table_df.sort_values(['data_quality','time_difference_of_start'], ascending=[False, True])
            long_table_df = long_table_df.drop_duplicates(subset=['object_id','attribute_id'], keep='first')

            # data_querying_info
            if (saving_details_for_data_querying_info is not None):
                data_timestamps_df = long_table_df.groupby('valid_time_start').aggregate({'valid_time_start':'first','object_id':pd.Series.nunique})
                data_timestamps_df.reset_index(inplace=True, drop=True)
                data_timestamps_df = data_timestamps_df.sort_values('valid_time_start')
                data_timestamps_df.index = data_timestamps_df['valid_time_start']
                data_timestamps_dict = data_timestamps_df['object_id'].to_dict()
                simulation_model = Simulation_model.objects.get(id=saving_details_for_data_querying_info['simulation_id'])
                data_querying_info = json.loads(simulation_model.data_querying_info)
                data_querying_info['timestamps'][saving_details_for_data_querying_info['object_number']] = data_timestamps_dict
                simulation_model.data_querying_info = json.dumps(data_querying_info)
                simulation_model.save()


            # pivot the long table
            print('2.11 - ' + str(len(long_table_df)))
            long_table_df = long_table_df.reindex()
            long_table_df = long_table_df[['object_id','satisfying_time_start','attribute_id', 'string_value', 'numeric_value','boolean_value' ]]
            long_table_df.set_index(['object_id','satisfying_time_start','attribute_id'],inplace=True)
            broad_table_df = long_table_df.unstack('attribute_id')

            # there are columns for the different datatypes, determine which to keep
            columns_to_keep = []
            print('2.12 - ' + str(len(broad_table_df)))
            for column in broad_table_df.columns:
                attribute_data_type = Attribute.objects.get(id=column[1]).data_type
                if attribute_data_type=='string' and column[0]=='string_value':
                    columns_to_keep.append(column)
                elif attribute_data_type in ['real', 'int', 'relation'] and column[0]=='numeric_value':
                    columns_to_keep.append(column)
                elif attribute_data_type == 'bool' and column[0]=='boolean_value':
                    columns_to_keep.append(column)


            broad_table_df = broad_table_df[columns_to_keep]
            new_column_names = [column[1] for column in columns_to_keep]
            broad_table_df.columns = new_column_names

            # clean up the broad table
            print('2.13 - ' + str(len(broad_table_df)))
            broad_table_df['object_id'] = [val[0] for val in broad_table_df.index]
            list_of_datetimes = [datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=(val[1])) for val in broad_table_df.index]
            broad_table_df['time'] = [val.strftime('%Y-%m-%d') for val in list_of_datetimes]

            broad_table_df.reindex()
            broad_table_df = broad_table_df.where(pd.notnull(broad_table_df), None)


            # insert the object_type's additional facts
            print('2.14 - ' + str(len(broad_table_df)))
            additional_attribute_values = []
            list_of_parent_object_types = [el['id'] for el in get_from_db.get_list_of_parent_objects(object_type_id)]
            li_attr_strings = list(Object_types.objects.filter(id__in=list_of_parent_object_types).values_list('li_attr'))
            for li_attr_string in li_attr_strings:
                li_attr = json.loads(li_attr_string[0])
                if 'attribute_values' in li_attr:
                    additional_attribute_values += li_attr['attribute_values']
            additional_attribute_values = [attribute_value for attribute_value in additional_attribute_values if attribute_value['operation']=='=']
            for attribute_value in additional_attribute_values:
                if isinstance(attribute_value['value'], int):
                    attribute_value['value'] = float(attribute_value['value'])
                broad_table_df[str(attribute_value['attribute_id'])] = attribute_value['value']


            # insert missing columns
            all_attribute_ids = Attribute.objects.filter(first_applicable_object_type__in = list_of_parent_object_types).values_list('id', flat=True)
            all_attribute_ids = [str(attribute_id) for attribute_id in all_attribute_ids]
            existing_columns = list(broad_table_df.columns)
            for attribute_id in all_attribute_ids:
                if attribute_id not in existing_columns:
                    broad_table_df[attribute_id] = None

            return broad_table_df









def get_objects_true_timeline(object_id, simulated_timeline_df):

    true_timeline_df = simulated_timeline_df.copy()
    error_timeline_df = simulated_timeline_df.copy()
    attribute_ids = [column for column in list(simulated_timeline_df.columns) if column not in ['start_time','end_time']]
    attribute_data_types = {}
    for attribute_id in attribute_ids:
        attribute_data_types[attribute_id] = Attribute.objects.get(id=attribute_id).data_type
        true_timeline_df[attribute_id] = np.nan
        error_timeline_df[attribute_id] = np.nan

    for index, row in true_timeline_df.iterrows():

        if index == 0:
            true_timeline_df.loc[index, attribute_id] = simulated_timeline_df.loc[index, attribute_id] 
            error_timeline_df.loc[index, attribute_id] = 0

        else:
            for attribute_id in attribute_ids:
                true_datapoint = Data_point.objects.filter(object_id=object_id, 
                                                            attribute_id=attribute_id, 
                                                            valid_time_start__lte=row['start_time'], 
                                                            valid_time_end__gt=row['start_time']).order_by('-data_quality', '-valid_time_start').first()

                simulated_value = simulated_timeline_df.loc[index, attribute_id]

                if attribute_data_types[attribute_id]=='string':
                    true_timeline_df.loc[index, attribute_id] = true_datapoint.string_value
                    error_timeline_df.loc[index, attribute_id] = 1 if true_datapoint.string_value.lower() == simulated_value.lower() else 0

                elif attribute_data_types[attribute_id] in ['real', 'int', 'relation']:
                    true_timeline_df.loc[index, attribute_id] = true_datapoint.numeric_value
                    true_increase = true_timeline_df.loc[index, attribute_id] - true_timeline_df.loc[index-1, attribute_id]
                    simulated_increase = simulated_timeline_df.loc[index, attribute_id] - simulated_timeline_df.loc[index-1, attribute_id]
                    error_value = abs(simulated_increase - true_increase) / true_increase
                    error_timeline_df.loc[index, attribute_id] = min(error_value, 1)

                elif attribute_data_types[attribute_id] == 'boolean':
                    true_timeline_df.loc[index, attribute_id] = true_datapoint.boolean_value
                    error_timeline_df.loc[index, attribute_id] = 1 if true_datapoint.boolean_value == simulated_value else 0

    return (true_timeline_df, error_timeline_df)




 # ===================================================================================================================
 #   ____  _     _                 _                             _   ______                _   _                 
 #  / __ \| |   | |               | |                           | | |  ____|              | | (_)                
 # | |  | | | __| |    _ __   ___ | |_ ______ _   _ ___  ___  __| | | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
 # | |  | | |/ _` |   | '_ \ / _ \| __|______| | | / __|/ _ \/ _` | |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
 # | |__| | | (_| |_  | | | | (_) | |_       | |_| \__ \  __/ (_| | | |  | |_| | | | | (__| |_| | (_) | | | \__ \
 #  \____/|_|\__,_( ) |_| |_|\___/ \__|       \__,_|___/\___|\__,_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
 #                |/                                                                                             
 # ===================================================================================================================








def find_matching_entities_OLD(match_attributes, match_values):
    matching_objects_entire_list = []

    for row_nb in range(len(match_values[0])):    
        matching_objects_dict = {}

        # append all matching datapoints
        found_datapoints = Data_point.objects.none()
        for attribute_nb, attribute_details in enumerate(match_attributes):
                
            additional_datapoints = Data_point.objects.filter(attribute_id=attribute_details['attribute_id'], value_as_string=str(match_values[attribute_nb][row_nb]))
            found_datapoints = found_datapoints.union(additional_datapoints)
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            # print(attribute_details['attribute_id'])
            # print(str(match_values[attribute_nb][row_nb]))
            # print(len(found_datapoints)) 
            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

        # get the object_ids  - those with most matching datapoints first
        found_datapoints_df = pd.DataFrame(list(found_datapoints.values('object_id','attribute_id','value_as_string','data_quality')))
        if len(found_datapoints_df)==0:
            matching_objects_entire_list.append([])
        else:
            found_object_attributes_df = found_datapoints_df.groupby(['object_id','attribute_id','value_as_string']).aggregate({'object_id': 'first','attribute_id': 'first', 'value_as_string':'first', 'data_quality':np.sum})
            object_scores_df = found_object_attributes_df.groupby('object_id').aggregate({'object_id':'first', 'attribute_id':'count', 'data_quality':np.sum})
            

            objects_df = found_object_attributes_df.pivot(index='object_id', columns='attribute_id', values='value_as_string')
            objects_df['object_id'] = objects_df.index
            objects_df = pd.merge(objects_df, object_scores_df, on='object_id', how='left')
            objects_df = objects_df.sort_values(['attribute_id','data_quality'], ascending=[False, False])
            objects_df = objects_df[:3]
            object_columns = list(objects_df.columns)
            attribute_ids = [attribute['attribute_id'] for attribute in match_attributes if (attribute['attribute_id'] in object_columns)]
            objects_df = objects_df[['object_id'] + attribute_ids]
            matching_objects_json = objects_df.to_json(orient='records')
            if matching_objects_json is not None:
                matching_objects_entire_list.append(json.loads(matching_objects_json))

    return matching_objects_entire_list















def filter_and_make_df_from_datapoints_OLD(object_type_id, object_ids, filter_facts, specified_start_time, specified_end_time):

    data_point_records = Data_point.objects.filter(object_id__in=object_ids)
    data_point_records = data_point_records.filter(valid_time_start__gte=specified_start_time, valid_time_start__lt=specified_end_time)   
    object_ids = list(set(list(data_point_records.values_list('object_id', flat=True).distinct())))


    # apply filter-facts
    valid_ranges_df = pd.DataFrame({'object_id':object_ids})
    valid_ranges_df['valid_range'] = [[[specified_start_time,specified_end_time]] for i in valid_ranges_df.index]

    for filter_fact in filter_facts:
        if filter_fact['operation'] == '=':
            filtered_data_point_records = data_point_records.filter(attribute_id=filter_fact['attribute_id'],string_value=str(filter_fact['value']))           
        elif filter_fact['operation'] == '>':
            filtered_data_point_records = data_point_records.filter(attribute_id=filter_fact['attribute_id'],numeric_value__gt=filter_fact['value'])           
        elif filter_fact['operation'] == '<':
            filtered_data_point_records = data_point_records.filter(attribute_id=filter_fact['attribute_id'],numeric_value__lt=filter_fact['value'])           
        elif filter_fact['operation'] == 'in':
            values = [str(value) for value in filter_fact['value']]
            filtered_data_point_records = data_point_records.filter(attribute_id=filter_fact['attribute_id'],string_value__in=values)           

        # find the intersecting time ranges (= the overlap between the known valid_ranges and the valid_ranges from the new filter fact)
        filtered_data_points_df = pd.DataFrame(list(filtered_data_point_records.values('object_id','valid_time_start','valid_time_end')))
        if len(filtered_data_points_df) == 0:
            return None
        filtered_data_points_df = filtered_data_points_df.sort_values(['object_id','valid_time_start'])
        filtered_data_points_df['new_valid_range'] = list(zip(filtered_data_points_df['valid_time_start'],filtered_data_points_df['valid_time_end']))
        new_valid_ranges_df = pd.DataFrame(filtered_data_points_df.groupby('object_id')['new_valid_range'].apply(list)) # for every object_id there now is a list of valid ranges
        new_valid_ranges_df['object_id']  = new_valid_ranges_df.index

        valid_ranges_df = pd.merge(valid_ranges_df, new_valid_ranges_df, on='object_id', how='left')
        valid_ranges_df = valid_ranges_df[valid_ranges_df['new_valid_range'].notnull()]
        if len(valid_ranges_df) == 0:
            return None
        valid_ranges_df['valid_range'] = valid_ranges_df.apply(generally_useful_functions.intersections, axis=1)
        # valid_ranges_df['valid_range'] = np.vectorize(generally_useful_functions.intersections)(valid_ranges_df['valid_range'], valid_ranges_df['new_valid_range'])
        valid_ranges_df = valid_ranges_df[['object_id', 'valid_range']]

    # choose the first time interval that satisfies all filter-fact conditions
    valid_ranges_df['satisfying_time_start'] = [object_ranges[0][0] if len(object_ranges)>0 else None for object_ranges in valid_ranges_df['valid_range'] ]
    valid_ranges_df['satisfying_time_end'] = [object_ranges[0][1] if len(object_ranges)>0 else None for object_ranges in valid_ranges_df['valid_range']]
    valid_ranges_df = valid_ranges_df[valid_ranges_df['satisfying_time_start'].notnull()]

    # make long table with all datapoints of the found objects
    found_objects = list(set(data_point_records.values_list('object_id', flat=True)))
    if len(found_objects) == 0:
        return None
    all_data_points = Data_point.objects.filter(object_id__in=found_objects)    
    long_table_df = pd.DataFrame(list(all_data_points.values()))

    # filter out the observations from not-satisfying times
    long_table_df = pd.merge(long_table_df, valid_ranges_df, how='inner', on='object_id')
    long_table_df = long_table_df[(long_table_df['valid_time_end'] > long_table_df['satisfying_time_start']) & (long_table_df['valid_time_start'] < long_table_df['satisfying_time_end'])]


    # select satisfying time (and remove the records from other times)
    total_data_quality_df = long_table_df.groupby(['object_id','satisfying_time_start']).aggregate({'object_id':'first','satisfying_time_start':'first', 'data_quality': np.sum, 'attribute_id': 'count'})
    total_data_quality_df = total_data_quality_df.rename(columns={"data_quality": "total_data_quality", "attribute_id":"attriubte_count"})

    total_data_quality_df = total_data_quality_df.sort_values(['total_data_quality','satisfying_time_start'], ascending=[False, True])
    total_data_quality_df = total_data_quality_df.drop_duplicates(subset=['object_id'], keep='first')
    long_table_df = pd.merge(long_table_df, total_data_quality_df, how='inner', on=['object_id','satisfying_time_start'])

    # remove the duplicates (=duplicate values within the satisfying time)
    long_table_df['time_difference_of_start'] = abs(long_table_df['satisfying_time_start'] - long_table_df['valid_time_start'])
    long_table_df = long_table_df.sort_values(['data_quality','time_difference_of_start'], ascending=[False, True])
    long_table_df = long_table_df.drop_duplicates(subset=['object_id','attribute_id'], keep='first')

    # pivot the long table
    long_table_df = long_table_df.reindex()
    long_table_df = long_table_df[['object_id','satisfying_time_start','attribute_id', 'string_value', 'numeric_value','boolean_value' ]]
    long_table_df.set_index(['object_id','satisfying_time_start','attribute_id'],inplace=True)
    broad_table_df = long_table_df.unstack('attribute_id')

    # there are columns for the different datatypes, determine which to keep
    columns_to_keep = []
    for column in broad_table_df.columns:
        attribute_data_type = Attribute.objects.get(id=column[1]).data_type
        if attribute_data_type=='string' and column[0]=='string_value':
            columns_to_keep.append(column)
        elif attribute_data_type in ['real', 'int','relation'] and column[0]=='numeric_value':
            columns_to_keep.append(column)
        elif attribute_data_type == 'boolean' and column[0]=='boolean_value':
            columns_to_keep.append(column)

    broad_table_df = broad_table_df[columns_to_keep]
    new_column_names = [column[1] for column in columns_to_keep]
    broad_table_df.columns = new_column_names

    # clean up the broad table
    broad_table_df['object_id'] = [val[0] for val in broad_table_df.index]
    broad_table_df['time'] = [datetime.datetime.fromtimestamp(val[1]).strftime('%Y-%m-%d') for val in broad_table_df.index] # this is the chosen satisfying_time_start for the object_id
    broad_table_df.reindex()
    broad_table_df = broad_table_df.where(pd.notnull(broad_table_df), None)
    
    return broad_table_df


