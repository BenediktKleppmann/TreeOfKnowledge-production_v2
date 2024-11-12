####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (c) 2019-2040 Benedikt Kleppmann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version - see http://www.gnu.org/licenses/.
#####################################################################


from collection.models import Object_types, Attribute, Data_point, Object, Uploaded_dataset
import pandas as pd
import json
import datetime
import os
from django.db.models import Count, Max
from django.db import connection
from collection.functions import get_from_db
 


 # ==================================================================
 #  _____                  _       _         _____  ____  
 # |  __ \                | |     | |       |  __ \|  _ \ 
 # | |__) |__  _ __  _   _| | __ _| |_ ___  | |  | | |_) |
 # |  ___/ _ \| '_ \| | | | |/ _` | __/ _ \ | |  | |  _ < 
 # | |  | (_) | |_) | |_| | | (_| | ||  __/ | |__| | |_) |
 # |_|   \___/| .__/ \__,_|_|\__,_|\__\___| |_____/|____/ 
 #            | |                                         
 #            |_|                                         
 # ==================================================================


# Clear --------------------------------------------------

def clear_object_types():
    Object_types.objects.all().delete()


def clear_attributes():
    Attribute.objects.all().delete()



# Populate --------------------------------------------------

def populate_object_types():

    # You could alternatively run the following in the command line
    #       python manage.py dumpdata collection.object_types > "collection/static/webservice files/db_backup/object_types.json"


    path = "collection/static/webservice files/db_backup/object_types/"
    backup_files = os.listdir(path)
    with open(path + backup_files[-1], "r") as backup_file:
        lines = backup_file.readlines()

    records_dict = json.loads(lines[0])
    for record in records_dict:
        if isinstance(record,dict):
            obj_type_record = Object_types( id=record.get('id'), 
                                            parent=record.get('parent'), 
                                            name=record.get('text'), 
                                            li_attr=json.dumps(record.get('li_attr')), 
                                            a_attr=json.dumps(record.get('a_attr')),)
            obj_type_record.save()

def populate_attributes():

    path = "collection/static/webservice files/db_backup/attributes/"
    backup_files = os.listdir(path)
    with open(path + backup_files[-1], "r") as backup_file:
        lines = backup_file.readlines()

    attributes = json.loads(lines[0])

    print("------------------------------------------")
    print("------------------------------------------")
    print(attributes)
    print("------------------------------------------")
    print("------------------------------------------")
    for attribute in attributes:
        attribute_record = Attribute(id=attribute['id'], 
                                    name=attribute['name'], 
                                    expected_valid_period=attribute['expected_valid_period'], 
                                    description=attribute['description'], 
                                    format_specification=json.dumps(attribute['format_specification']), 
                                    first_applicable_object_type=attribute['first_applicable_object_type'],
									first_relation_object_type=attribute['first_relation_object_type'],)
        attribute_record.save()



# Backup DB --------------------------------------------------

def backup_object_types():
    result_list = []
    object_types = Object_types.objects.all()
    for object_type in object_types:
        object_type_dict = {'id':object_type.id, 
                            'parent':object_type.parent, 
                            'text':object_type.name}
        
        if object_type.li_attr is not None:
            object_type_dict['li_attr'] = json.loads(object_type.li_attr)

        if object_type.a_attr is not None:
            object_type_dict['a_attr'] = json.loads(object_type.a_attr)

        result_list.append(object_type_dict)

    file_path = "collection/static/webservice files/db_backup/object_types/" + str(datetime.datetime.now()).replace(':','') + ".json"
    # file_path = str(datetime.datetime.now()).replace(':','') + ".json"
    with open(file_path, "w") as file:
            file.write(json.dumps(result_list))
    return True


def backup_attributes():
    result_list = []
    attributes = Attribute.objects.all()
    for attribute in attributes:
        result_list.append({'id':attribute.id, 
                            'name':attribute.name, 
                            'description':attribute.description, 
                            'expected_valid_period':attribute.expected_valid_period, 
                            'format_specification':json.loads(attribute.format_specification), 
                            'first_applicable_object_type':attribute.first_applicable_object_type,
							'first_relation_object_type':attribute.first_relation_object_type})

    file_path = "collection/static/webservice files/db_backup/attributes/" + str(datetime.datetime.now()).replace(':','') + ".json"
    # file_path = str(datetime.datetime.now()).replace(':','') + ".json"
    with open(file_path, "w") as file:
        file.write(json.dumps(result_list))
    return True



 # ====================================================================
 #  _____                           _     _____        _        
 # |_   _|                         | |   |  __ \      | |       
 #   | |  _ __  ___ _ __   ___  ___| |_  | |  | | __ _| |_ __ _ 
 #   | | | '_ \/ __| '_ \ / _ \/ __| __| | |  | |/ _` | __/ _` |
 #  _| |_| | | \__ \ |_) |  __/ (__| |_  | |__| | (_| | || (_| |
 # |_____|_| |_|___/ .__/ \___|\___|\__| |_____/ \__,_|\__\__,_|
 #                 | |                                          
 #                 |_|                                          
 # ====================================================================

def inspect_individual_object(object_id):

    sql_query = 'SELECT object_id, attribute_id, valid_time_start, valid_time_end,  value_as_string, upload_id FROM collection_data_point WHERE object_id = %s' % (object_id)
    object_values_df = pd.read_sql_query(sql_query, connection)

    # attribute name mappings
    attributes = list(Attribute.objects.all().values())
    attributes_dict = {str(attribute['id']):attribute['name'] for attribute in attributes}

    object_values_df['attribute_name'] = object_values_df['attribute_id'].replace(attributes_dict)
    object_values_df['valid_time_start'] = pd.to_datetime(object_values_df['valid_time_start'], unit='s').dt.strftime('%Y-%m-%d')
    object_values_df['valid_time_end'] = pd.to_datetime(object_values_df['valid_time_end'], unit='s').dt.strftime('%Y-%m-%d')
    object_values_df['Attribute and Time'] = object_values_df['attribute_name'] + ' (' + object_values_df['valid_time_start'] + ' - ' + object_values_df['valid_time_end'] + ')'
    object_values_df = object_values_df.rename(columns={"value_as_string": "Value"})

    upload_ids = list(object_values_df['upload_id'])
    print('=========================================================')
    print(object_values_df.columns)
    print('=========================================================')
    object_values_df = object_values_df[['Attribute and Time', 'Value']]

    # object_dict
    object_type_id = Object.objects.get(id=object_id).object_type_id
    object_type = Object_types.objects.get(id=object_type_id).name
    object_dict = {'table_headers':list(object_values_df.columns), 'table_data':object_values_df.values.tolist(),'object_type_id':object_type_id, 'object_type':object_type, 'upload_ids':upload_ids}
    return object_dict



# ====================================================================
#    _____ _                    _____        _        
#   / ____| |                  |  __ \      | |       
#  | |    | | ___  __ _ _ __   | |  | | __ _| |_ __ _ 
#  | |    | |/ _ \/ _` | '_ \  | |  | |/ _` | __/ _` |
#  | |____| |  __/ (_| | | | | | |__| | (_| | || (_| |
#   \_____|_|\___|\__,_|_| |_| |_____/ \__,_|\__\__,_|
#
# ====================================================================                                                    
                                                   





def remove_datapoints_with_the_wrong_datatype():
    numeric_attribute_ids = list(Attribute.objects.filter(data_type__in=['real', 'int', 'relation']).values_list('id', flat=True))
    numeric_violating_datapoints = Data_point.objects.filter(attribute_id__in=numeric_attribute_ids).filter(numeric_value__isnull=True)
    numeric_violating_datapoints.delete()

    boolean_attribute_ids = list(Attribute.objects.filter(data_type='boolean').values_list('id', flat=True))
    boolean_violating_datapoints = Data_point.objects.filter(attribute_id__in=boolean_attribute_ids).filter(boolean_value__isnull=True)
    boolean_violating_datapoints.delete()

    string_attribute_ids = list(Attribute.objects.filter(data_type='string').values_list('id', flat=True))
    string_violating_datapoints = Data_point.objects.filter(attribute_id__in=string_attribute_ids).filter(string_value__isnull=True)
    string_violating_datapoints.delete()

    return 'success'



# ===========================================================================
# NULL Values ================================================================
# ===========================================================================

def remove_null_datapoints():
    print('admin_functions.remove_null_datapoints')
    with connection.cursor() as cursor:
        print('1')
        cursor.execute('''DELETE FROM  collection_data_point
                          WHERE value_as_string in ('null', 'nan', 'None', 'unknown');
                                ''')
        print('2')




# ===========================================================================
# Duplicates ================================================================
# ===========================================================================

def remove_duplicates():
    unique_fields = ['object_id', 'attribute_id', 'value_as_string', 'numeric_value', 'string_value', 'boolean_value', 'valid_time_start', 'valid_time_end', 'data_quality']

    duplicates = (
        Data_point.objects.values(*unique_fields)
        .order_by()
        .annotate(max_id=Max('id'), count_id=Count('id'))
        .filter(count_id__gt=1)
    )

    for duplicate in duplicates:
        (
            Data_point.objects
            .filter(**{x: duplicate[x] for x in unique_fields})
            .exclude(id=duplicate['max_id'])
            .delete()
    )
        


def find_possibly_duplicate_objects_without_keys():

    with open("collection/static/webservice files/runtime_data/duplicate_objects_by_object_type.txt", "w") as text_file:
        text_file.write('Notice: The finding of possible duplicates is in progress.')

    duplicate_objects_by_object_type = {}


    # if 'DATABASE_URL' in dict(os.environ).keys() and dict(os.environ)['DATABASE_URL'][:8]!='postgres':
    if 'IS_USING_SQLITE_DB' in dict(os.environ).keys():
        # SQLITE...
        with connection.cursor() as cursor:
            print('5')
            sql_string2 = '''
                SELECT 
                    group_concat(objects.object_id) as object_ids
                FROM 
                (
                    SELECT ordered_data_points.object_id,
                            group_concat(ordered_data_points.value_as_string) as concatenated_values
                    FROM
                    (
                        SELECT *
                        FROM collection_data_point 
                        ORDER BY object_id, attribute_id, valid_time_start, valid_time_end, value_as_string
                    ) ordered_data_points
                    GROUP BY ordered_data_points.object_id
                ) objects
                GROUP BY objects.concatenated_values
                HAVING COUNT(*) > 1
                '''
            print('6')
            cursor.execute(sql_string2)
            print('7')
            result = cursor.fetchall()
            print('8')
            list_of_lists__duplicate_objects = list(result)
            print('9')

    else:
        # POSTGES... 
        with connection.cursor() as cursor:
            print('1')
            sql_string2 = '''
                SELECT 
                    string_agg(CAST(objects.object_id AS TEXT), ',') as object_ids
                FROM 
                (
                    SELECT ordered_data_points.object_id,
                            string_agg(ordered_data_points.value_as_string, ',') as concatenated_values
                    FROM
                    (
                        SELECT *
                        FROM collection_data_point 
                        ORDER BY object_id, attribute_id, valid_time_start, valid_time_end, value_as_string
                    ) ordered_data_points
                    GROUP BY ordered_data_points.object_id
                ) objects
                GROUP BY objects.concatenated_values
                HAVING COUNT(*) > 1
                '''
            print('2')
            cursor.execute(sql_string2)
            print('3')
            result = cursor.fetchall()
            print('4')
            list_of_lists__duplicate_objects = list(result)

       



    all_duplicate_objects = []
    print('10')
    for entry_nb, duplicate_objects in enumerate(list_of_lists__duplicate_objects):
        print()


        # name mappings
        print('11')
        attributes = list(Attribute.objects.all().values())
        attributes_dict = {str(attribute['id']):attribute['name'] for attribute in attributes}
        

        # table_data
        print('12')
        duplicate_objects_str = [str(object_id) for object_id in duplicate_objects]
        sql_query = 'SELECT object_id, attribute_id, valid_time_start, valid_time_end,  value_as_string FROM collection_data_point WHERE object_id IN (%s)' % (','.join(duplicate_objects_str))
        duplicate_object_values_df = pd.read_sql_query(sql_query, connection)

        duplicate_object_values_df['attribute_name'] = duplicate_object_values_df['attribute_id'].replace(attributes_dict)
        duplicate_object_values_df['valid_time_start'] = pd.to_datetime(duplicate_object_values_df['valid_time_start']).dt.strftime('%Y-%m-%d')
        duplicate_object_values_df['valid_time_end'] = pd.to_datetime(duplicate_object_values_df['valid_time_end']).dt.strftime('%Y-%m-%d')
        duplicate_object_values_df['attribute_and_time'] = duplicate_object_values_df['attribute_name'] + ' (' + duplicate_object_values_df['valid_time_start'] + ' - ' + duplicate_object_values_df['valid_time_end'] + ')'
        duplicate_object_values_df = duplicate_object_values_df[['object_id', 'attribute_and_time', 'value_as_string']]
        duplicate_object_values_df = duplicate_object_values_df.drop_duplicates(['object_id','attribute_and_time'])
        duplicate_objects_df = duplicate_object_values_df.pivot(index='object_id', columns='attribute_and_time', values='value_as_string')

        table_headers = list(duplicate_objects_df.columns)
        table_data = duplicate_objects_df.values.tolist()


        # deletable_objects
        print('13')
        object_ids = json.loads('['+ str(duplicate_objects[0]) + ']')

        # object_types
        object_type_id = Object.objects.get(id=object_ids[0]).object_type_id


        object_dict = {'table_headers':table_headers, 'table_data':table_data, 'object_ids':object_ids}
        if object_type_id not in duplicate_objects_by_object_type:
            duplicate_objects_by_object_type[object_type_id] = [object_dict]
        else:
            duplicate_objects_by_object_type[object_type_id].append(object_dict)

        with open("collection/static/webservice files/runtime_data/duplicate_objects_by_object_type.txt", "w") as text_file:
            text_file.write(json.dumps(duplicate_objects_by_object_type))

    return duplicate_objects_by_object_type







def get_possibly_duplicate_objects_with_keys(object_type_id, key_attribute_id):

    # basic filtering: object_type and specified time range
    child_object_types = get_from_db.get_list_of_child_objects(object_type_id)
    child_object_ids = [el['id'] for el in child_object_types]

    with connection.cursor() as cursor:
        sql_string1 = 'SELECT DISTINCT id FROM collection_object WHERE object_type_id IN (%s);' % (', '.join("'{0}'".format(object_type_id) for object_type_id in child_object_ids))
        cursor.execute(sql_string1)
        object_ids = [str(entry[0]) for entry in cursor.fetchall()]




     # if 'DATABASE_URL' in dict(os.environ).keys() and dict(os.environ)['DATABASE_URL'][:8]!='postgres':
    if 'IS_USING_SQLITE_DB' in dict(os.environ).keys():
        # SQLITE...
        with connection.cursor() as cursor:
            sql_string2 = '''
                SELECT  group_concat(object_id) as object_ids
                FROM collection_data_point 
                WHERE attribute_id = %s
                AND object_id IN (%s)
                GROUP BY value_as_string
                HAVING COUNT(DISTINCT object_id) > 1
                '''

            cursor.execute(sql_string2 % (str(key_attribute_id), ','.join(object_ids)))
            result = cursor.fetchall()
            list_of_lists__duplicate_objects = list(result)


    else:
        # POSTGES... 
        with connection.cursor() as cursor:
            print('1')
            sql_string2 = '''
                SELECT  string_agg(CAST(object_id AS TEXT), ',') as object_ids
                FROM collection_data_point 
                WHERE attribute_id = %s
                AND object_id IN (%s)
                GROUP BY value_as_string
                HAVING COUNT(DISTINCT object_id) > 1
                '''

            cursor.execute(sql_string2 % (str(key_attribute_id), ','.join(object_ids)))
            result = cursor.fetchall()
            list_of_lists__duplicate_objects = list(result)


    possibly_duplicate_objects = []
    for entry_nb, duplicate_objects in enumerate(list_of_lists__duplicate_objects):

        # name mappings
        print('11')
        attributes = list(Attribute.objects.all().values())
        attributes_dict = {str(attribute['id']):attribute['name'] for attribute in attributes}
        

        # table_data
        print('12')
        duplicate_objects_str = [str(object_id) for object_id in duplicate_objects]
        sql_query = 'SELECT object_id, attribute_id, valid_time_start, valid_time_end,  value_as_string FROM collection_data_point WHERE object_id IN (%s)' % (','.join(duplicate_objects_str))
        duplicate_object_values_df = pd.read_sql_query(sql_query, connection)

        duplicate_object_values_df['attribute_name'] = duplicate_object_values_df['attribute_id'].replace(attributes_dict)
        duplicate_object_values_df['valid_time_start'] = pd.to_datetime(duplicate_object_values_df['valid_time_start']).dt.strftime('%Y-%m-%d')
        duplicate_object_values_df['valid_time_end'] = pd.to_datetime(duplicate_object_values_df['valid_time_end']).dt.strftime('%Y-%m-%d')
        duplicate_object_values_df['attribute_and_time'] = duplicate_object_values_df['attribute_name'] + ' (' + duplicate_object_values_df['valid_time_start'] + ' - ' + duplicate_object_values_df['valid_time_end'] + ')'
        duplicate_object_values_df = duplicate_object_values_df[['object_id', 'attribute_and_time', 'value_as_string']]
        duplicate_object_values_df = duplicate_object_values_df.drop_duplicates(['object_id','attribute_and_time'])
        duplicate_objects_df = duplicate_object_values_df.pivot(index='object_id', columns='attribute_and_time', values='value_as_string') 
        duplicate_objects_df = duplicate_objects_df.where((pd.notnull(duplicate_objects_df)), None) # replace 'np.nan' with 'None', because json.dumps can only handle 'None'


        table_headers = list(duplicate_objects_df.columns)
        table_data = duplicate_objects_df.values.tolist()


        # deletable_objects
        print('13')
        object_ids = json.loads('['+ str(duplicate_objects[0]) + ']')


        object_dict = {'table_headers':table_headers, 'table_data':table_data, 'object_ids':object_ids}
        possibly_duplicate_objects.append(object_dict)

    return possibly_duplicate_objects





