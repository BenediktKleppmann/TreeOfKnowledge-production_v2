####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (C) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, February 2021
#####################################################################

from collection.models import Uploaded_dataset, Object_types, Attribute, Object, Likelihood_function, Execution_order
from django.db.models import Count
import json
import numpy as np
import pdb
import pandas as pd
import scipy
import scipy.stats as stats





def get_object_hierachy_tree():
    hierachy_objects = Object_types.objects.all()

    object_hierachy_tree = []
    for hierachy_object in hierachy_objects:
        object_dict = {}
        object_dict['id'] = hierachy_object.id
        object_dict['parent'] = hierachy_object.parent
        object_dict['text'] = hierachy_object.name
        if hierachy_object.li_attr is not None:
            print("UUUUUUUUUUUUUUUUUUUUUUUUUUUUUU")
            print(hierachy_object.name)
            print(hierachy_object.id)
            print(hierachy_object.li_attr)
            print(hierachy_object.a_attr)
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            object_dict['li_attr'] = json.loads(hierachy_object.li_attr)
        if hierachy_object.a_attr is not None:
            object_dict['a_attr'] = json.loads(hierachy_object.a_attr)
        
        object_hierachy_tree.append(object_dict)
    object_hierachy_tree_json = json.dumps(object_hierachy_tree)
    return object_hierachy_tree_json
    # return [{"id":"n1", "text" : "Thing", "li_attr": {}, "parent":"#"}, {"id":"n2", "text" : "Object" , "li_attr": {}, "parent":"n1"}, {"id":"n3", "text": "Living thing", "li_attr": {}, "parent":"n2"}, {"id":"n4", "text": "Plant", "li_attr": {"attribute_values": [{"attribute":"Kingdom", "operation":"=", "value":"Plantae"}, {"attribute":"Does photosynthesis", "operation": "=", "value":true}]}, "a_attr":{"scientific":["Plantae"]}, "parent":"n3"}, {"id":"n5", "text": "Tree", "li_attr": {"attribute_values": [{"attribute":"Has woody tissue", "operation": "=", "value":true},{"attribute":"Age", "operation": "<", "value":"7000"}]},  "parent":"n4"}, {"id":"n6", "text": "Oak", "li_attr": {"attribute_values": [{"attribute":"Produces nuts", "operation": "=", "value":true}, {"attribute":"Has leaves", "operation": "=", "value":true}, {"attribute":"Age", "operation": "<", "value":"700"}, {"attribute":"Age", "operation": "<", "value":"100"}, {"attribute":"Weight", "operation": "<", "value":"9000"}]}, "a_attr":{"synonyms": ["oak tree"], "scientific": ["Quercus"]}, "parent":"n5"}, {"id":"n7", "text": "Chestnut", "li_attr": {"attribute_values": [{"attribute":"Produces nuts", "operation": "=", "value":true}, {"attribute":"Produces berries", "operation": "=", "value":false}, {"attribute":"Age", "operation": "<", "value":"400"}, {"attribute":"Height", "operation": "<", "value":"130"}, {"attribute":"Weight", "operation": "<", "value":"10000"}]}, "a_attr":{"scientific": ["Castanea"]}, "parent":"n5"}, {"id":"n8", "text": "Flower", "li_attr": {"attribute_values": [{"attribute":"Has petals", "operation": "=", "value":true}]}, "parent":"n4"}, {"id":"n9", "text": "Lily", "li_attr": {"attribute_values": [{"attribute":"Petal color", "operation": "=", "value":"yellow"}]}, "parent":"n8"}, {"id":"n10", "text": "Animal", "li_attr": {"attribute_values": [{"attribute":"Kingdom", "operation": "=", "value":"Animalia"}]}, "a_attr":{"synonyms": ["Creature"], "scientific": ["Animalia"]}, "parent":"n2"} ]




def get_known_data_sources():
    return json.dumps(list(Uploaded_dataset.objects.order_by().values_list('data_source', flat=True).distinct()))
    # return ['3 Round Stones, Inc.', '48 Factoring Inc.', '5PSolutions', 'Abt Associates', 'Accela', 'Accenture', 'AccuWeather', 'Acxiom', 'Adaptive', 'Adobe Digital Government', 'Aidin', 'Alarm.com', 'Allianz', 'Allied Van Lines', 'Alltuition', 'Altova', 'Amazon Web Services', 'American Red Ball Movers', 'Amida Technology Solutions', 'Analytica', 'Apextech LLC', 'Appallicious', 'Aquicore', 'Archimedes Inc.', 'AreaVibes Inc.', 'Arpin Van Lines', 'Arrive Labs', 'ASC Partners', 'Asset4', 'Atlas Van Lines', 'AtSite', 'Aunt Bertha, Inc.', 'Aureus Sciences (*Now part of Elsevier)', 'AutoGrid Systems', 'Avalara', 'Avvo', 'Ayasdi', 'Azavea', 'BaleFire Global', 'Barchart', 'Be Informed', 'Bekins', 'Berkery Noyes MandASoft', 'Berkshire Hathaway', 'BetterLesson', 'BillGuard', 'Bing', 'Biovia', 'BizVizz', 'BlackRock', 'Bloomberg', 'Booz Allen Hamilton', 'Boston Consulting Group', 'Boundless', 'Bridgewater', 'Brightscope', 'BuildFax', 'Buildingeye', 'BuildZoom', 'Business and Legal Resources', 'Business Monitor International', 'Calcbench, Inc.', 'Cambridge Information Group', 'Cambridge Semantics', 'CAN Capital', 'Canon', 'Capital Cube', 'Cappex', 'Captricity', 'CareSet Systems', 'Careset.com', 'CARFAX', 'Caspio', 'Castle Biosciences', 'CB Insights', 'Ceiba Solutions', 'Center for Responsive Politics', 'Cerner', 'Certara', 'CGI', 'Charles River Associates', 'Charles Schwab Corp.', 'Chemical Abstracts Service', 'Child Care Desk', 'Chubb', 'Citigroup', 'CityScan', 'CitySourced', 'Civic Impulse LLC', 'Civic Insight', 'Civinomics', 'Civis Analytics', 'Clean Power Finance', 'ClearHealthCosts', 'ClearStory Data', 'Climate Corporation', 'CliniCast', 'Cloudmade', 'Cloudspyre', 'Code for America', 'Code-N', 'Collective IP', 'College Abacus, an ECMC initiative', 'College Board', 'Communitech', 'Compared Care', 'Compendia Bioscience Life Technologies', 'Compliance and Risks', 'Computer Packages Inc', 'CONNECT-DOT LLC.', 'ConnectEDU', 'Connotate', 'Construction Monitor LLC', 'Consumer Reports', 'CoolClimate', 'Copyright Clearance Center', 'CoreLogic', 'CostQuest', 'Credit Karma', 'Credit Sesame', 'CrowdANALYTIX', 'Dabo Health', 'DataLogix', 'DataMade', 'DataMarket', 'Datamyne', 'DataWeave', 'Deloitte', 'DemystData', 'Department of Better Technology', 'Development Seed', 'Docket Alarm, Inc.', 'Dow Jones & Co.', 'Dun & Bradstreet', 'Earth Networks', 'EarthObserver App', 'Earthquake Alert!', 'Eat Shop Sleep', 'Ecodesk', 'eInstitutional', 'Embark', 'EMC', 'Energy Points, Inc.', 'Energy Solutions Forum', 'Enervee Corporation', 'Enigma.io', 'Ensco', 'Environmental Data Resources', 'Epsilon', 'Equal Pay for Women', 'Equifax', 'Equilar', 'Ernst & Young LLP', 'eScholar LLC.', 'Esri', 'Estately', 'Everyday Health', 'Evidera', 'Experian', 'Expert Health Data Programming, Inc.', 'Exversion', 'Ez-XBRL', 'Factset', 'Factual', 'Farmers', 'FarmLogs', 'Fastcase', 'Fidelity Investments', 'FindTheBest.com', 'First Fuel Software', 'FirstPoint, Inc.', 'Fitch', 'FlightAware', 'FlightStats', 'FlightView', 'Food+Tech Connect', 'Forrester Research', 'Foursquare', 'Fujitsu', 'Funding Circle', 'FutureAdvisor', 'Fuzion Apps, Inc.', 'Gallup', 'Galorath Incorporated', 'Garmin', 'Genability', 'GenoSpace', 'Geofeedia', 'Geolytics', 'Geoscape', 'GetRaised', 'GitHub', 'Glassy Media', 'Golden Helix', 'GoodGuide', 'Google Maps', 'Google Public Data Explorer', 'Government Transaction Services', 'Govini', 'GovTribe', 'Govzilla, Inc.', 'gRadiant Research LLC', 'Graebel Van Lines', 'Graematter, Inc.', 'Granicus', 'GreatSchools', 'GuideStar', 'H3 Biomedicine', 'Harris Corporation', 'HDScores, Inc', 'Headlight', 'Healthgrades', 'Healthline', 'HealthMap', 'HealthPocket, Inc.', 'HelloWallet', 'HERE', 'Honest Buildings', 'HopStop', 'Housefax', "How's My Offer?", 'IBM', 'ideas42', 'iFactor Consulting', 'IFI CLAIMS Patent Services', 'iMedicare', 'Impact Forecasting (Aon)', 'Impaq International', 'Import.io', 'IMS Health', 'InCadence', 'indoo.rs', 'InfoCommerce Group', 'Informatica', 'InnoCentive', 'Innography', 'Innovest Systems', 'Inovalon', 'Inrix Traffic', 'Intelius', 'Intermap Technologies', 'Investormill', 'Iodine', 'IPHIX', 'iRecycle', 'iTriage', 'IVES Group Inc', 'IW Financial', 'JJ Keller', 'J.P. Morgan Chase', 'Junar, Inc.', 'Junyo', 'Kaiser Permanante', 'karmadata', 'Keychain Logistics Corp.', 'KidAdmit, Inc.', 'Kimono Labs', 'KLD Research', 'Knoema', 'Knoema Corporation', 'Knowledge Agency', 'KPMG', 'Kroll Bond Ratings Agency', 'Kyruus', 'Lawdragon', 'Legal Science Partners', '(Leg)Cyte', 'LegiNation, Inc.', 'LegiStorm', 'Lenddo', 'Lending Club', 'Level One Technologies', 'LexisNexis', 'Liberty Mutual Insurance Cos.', 'Lilly Open Innovation Drug Discovery', 'Liquid Robotics', 'Locavore', 'LOGIXDATA, LLC', 'LoopNet', 'Loqate, Inc.', 'LoseIt.com', 'LOVELAND Technologies', 'Lucid', 'Lumesis, Inc.', 'Mango Transit', 'Mapbox', 'Maponics', 'MapQuest', 'Marinexplore, Inc.', 'MarketSense', 'Marlin & Associates', 'Marlin Alter and Associates', 'McGraw Hill Financial', 'McKinsey', 'MedWatcher', 'Mercaris', 'Merrill Corp.', 'Merrill Lynch', 'MetLife', 'mHealthCoach', 'MicroBilt Corporation', 'Microsoft Windows Azure Marketplace', 'Mint', "Moody's", 'Morgan Stanley', 'Morningstar, Inc.', 'Mozio', 'MuckRock.com', 'Munetrix', 'Municode', 'National Van Lines', 'Nationwide Mutual Insurance Company', 'Nautilytics', 'Navico', 'NERA Economic Consulting', 'NerdWallet', 'New Media Parents', 'Next Step Living', 'NextBus', 'nGAP Incorporated', 'Nielsen', 'Noesis', 'NonprofitMetrics', 'North American Van Lines', 'Noveda Technologies', 'NuCivic', 'Numedii', 'Oliver Wyman', 'OnDeck', 'OnStar', 'Ontodia, Inc', 'Onvia', 'Open Data Nation', 'OpenCounter', 'OpenGov', 'OpenPlans', 'OpportunitySpace, Inc.', 'Optensity', 'optiGov', 'OptumInsight', 'Orlin Research', 'OSIsoft', 'OTC Markets', 'Outline', 'Oversight Systems', 'Overture Technologies', 'Owler', 'Palantir Technologies', 'Panjiva', 'Parsons Brinckerhoff', 'Patently-O', 'PatientsLikeMe', 'Pave', 'Paxata', 'PayScale, Inc.', 'PeerJ', 'People Power', 'Persint', 'Personal Democracy Media', 'Personal, Inc.', 'Personalis', "Peterson's", 'PEV4me.com', 'PIXIA Corp', 'PlaceILive.com', 'PlanetEcosystems', 'PlotWatt', 'Plus-U', 'PolicyMap', 'Politify', 'Poncho App', 'POPVOX', 'Porch', 'PossibilityU', 'PowerAdvocate', 'Practice Fusion', 'Predilytics', 'PricewaterhouseCoopers (PWC)', 'ProgrammableWeb', 'Progressive Insurance Group', 'Propeller Health', 'ProPublica', 'PublicEngines', 'PYA Analytics', 'Qado Energy, Inc.', 'Quandl', 'Quertle', 'Quid', 'R R Donnelley', 'RAND Corporation', 'Rand McNally', 'Rank and Filed', 'Ranku', 'Rapid Cycle Solutions', 'realtor.com', 'Recargo', 'ReciPal', 'Redfin', 'RedLaser', 'Reed Elsevier', 'REI Systems', 'Relationship Science', 'Remi', 'Rentlogic', 'Retroficiency', 'Revaluate', 'Revelstone', 'Rezolve Group', 'Rivet Software', 'Roadify Transit', 'Robinson + Yu', 'Russell Investments', 'Sage Bionetworks', 'SAP', 'SAS', 'Scale Unlimited', 'Science Exchange', 'Seabourne', 'SeeClickFix', 'SigFig', 'Simple Energy', 'SimpleTuition', 'SlashDB', 'Smart Utility Systems', 'SmartAsset', 'SmartProcure', 'Smartronix', 'SnapSense', 'Social Explorer', 'Social Health Insights', 'SocialEffort Inc', 'Socrata', 'Solar Census', 'SolarList', 'Sophic Systems Alliance', 'S&P Capital IQ', 'SpaceCurve', 'SpeSo Health', 'Spikes Cavell Analytic Inc', 'Splunk', 'Spokeo', 'SpotCrime', 'SpotHero.com', 'Stamen Design', "Standard and Poor's", 'State Farm Insurance', 'Sterling Infosystems', 'Stevens Worldwide Van Lines', 'STILLWATER SUPERCOMPUTING INC', 'StockSmart', 'Stormpulse', 'StreamLink Software', 'StreetCred Software, Inc', 'StreetEasy', 'Suddath', 'Symcat', 'Synthicity', 'T. Rowe Price', 'Tableau Software', 'TagniFi', 'Telenav', 'Tendril', 'Teradata', 'The Advisory Board Company', 'The Bridgespan Group', 'The DocGraph Journal', 'The Govtech Fund', 'The Schork Report', 'The Vanguard Group', 'Think Computer Corporation', 'Thinknum', 'Thomson Reuters', 'TopCoder', 'TowerData', 'TransparaGov', 'TransUnion', 'TrialTrove', 'TrialX', 'Trintech', 'TrueCar', 'Trulia', 'TrustedID', 'TuvaLabs', 'Uber', 'Unigo LLC', 'United Mayflower', 'Urban Airship', 'Urban Mapping, Inc', 'US Green Data', 'U.S. News Schools', 'USAA Group', 'USSearch', 'Verdafero', 'Vimo', 'VisualDoD, LLC', 'Vital Axiom | Niinja', 'VitalChek', 'Vitals', 'Vizzuality', 'Votizen', 'Walk Score', 'WaterSmart Software', 'WattzOn', 'Way Better Patents', 'Weather Channel', 'Weather Decision Technologies', 'Weather Underground', 'WebFilings', 'Webitects', 'WebMD', 'Weight Watchers', 'WeMakeItSafer', 'Wheaton World Wide Moving', 'Whitby Group', 'Wolfram Research', 'Wolters Kluwer', 'Workhands', 'Xatori', 'Xcential', 'xDayta', 'Xignite', 'Yahoo', 'Zebu Compliance Solutions', 'Yelp', 'YourMapper', 'Zillow', 'ZocDoc', 'Zonability', 'Zoner', 'Zurich Insurance (Risk Room)']

def get_list_of_parent_objects(object_type_id):
    list_of_parent_objects = []
    current_object_type = Object_types.objects.filter(id=object_type_id).first()

    while(current_object_type is not None):
        list_of_parent_objects.append({'id':current_object_type.id, 'name':current_object_type.name})
        current_object_type = Object_types.objects.filter(id=current_object_type.parent).first()

    return list_of_parent_objects


def get_list_of_child_objects(object_type_id):
    # add the value from this generation
    current_generation = Object_types.objects.filter(id=object_type_id)
    list_of_child_objects = list(current_generation.values('id', 'name'))
    
    # successively add the values from the children 
    previous_generations_ids = [object_type_id]
    while len(previous_generations_ids)>0:
        current_generation = Object_types.objects.filter(parent__in=previous_generations_ids)
        list_of_child_objects += list(current_generation.values('id', 'name'))
        previous_generations_ids = current_generation.values_list('id', flat=True)

    return list_of_child_objects





# ================================================================
# COMPILED ATTRIBUTE FORMAT
# ================================================================

# this is just a helper function for get_from_db.get_attributes_concluding_format
def compare_facts_with_format_specification(list_of_facts, attribute_id, source_of_the_facts, format_specification, comments):
    for fact in list_of_facts:
            if (fact['attribute_id'] == attribute_id):
                if (format_specification['type'] in ['int', 'real', 'relation']) and (fact['operation'] in ['<', '>']):
                    # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
                    # print(format_specification)
                    # # print(type(fact['value']))
                    # # print(type(['max']))
                    # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    if(fact['operation'] == '<') and (fact['value'] < format_specification['max']):
                        # LOWER MAX FOUND
                        format_specification['max'] = fact['value']
                        comments['max'] +=  source_of_the_facts + ' -> ' + fact['attribute'] + ' ' + fact['operation'] + ' ' + str(fact['value']) + '.<br />'

                    if(fact['operation'] == '>') and (fact['value'] < format_specification['min']):
                        # LOWER MAX FOUND
                        format_specification['min'] = fact['value']
                        comments['min'] +=  source_of_the_facts + ' -> ' + fact['attribute'] + ' ' + fact['operation'] + ' ' + str(fact['value']) + '.<br />'

                if (format_specification['type'] == 'string') and (fact['operation'] == 'in'):
                    if ('allowed_values' in format_specification.keys()):
                        set_original = set(format_specification['allowed_values'])
                        set_fact = json.loads(fact['value'])
                        new_allowed_values = list(set.intersection(set1, set2))
                        if len(new_allowed_values) < format_specification['allowed_values']:
                            format_specification['allowed_values'] = new_allowed_values
                            comments['allowed_values'] += source_of_the_facts + ' -> ' + fact['attribute'] + ' ' + fact['operation'] + ' ' + str(fact['value']) + '.<br />'
                    else:
                        format_specification['allowed_values'] = json.loads(fact['value'])
                        comments['allowed_values'] += source_of_the_facts + ' -> ' + fact['attribute'] + ' ' + fact['operation'] + ' ' + str(fact['value']) + '.<br />'
    return (format_specification, comments)


def get_attributes_concluding_format(attribute_id, object_type_id, upload_id):
    print('4.1')
    list_of_parent_objects = get_list_of_parent_objects(object_type_id)
    list_of_parent_objects.reverse()

    print('4.2')
    attribute_record = Attribute.objects.get(id=attribute_id)
    constraint_dict =  json.loads(attribute_record.format_specification)
    format_specification  = constraint_dict['fields']['column']
    comments = {'min':'','max':'','allowed_values':''}
    
    # Additional Format Restrictions from the Object Type 
    print('4.3')
    for parent_object in list_of_parent_objects:
        parent_object_record = Object_types.objects.get(id=parent_object['id'])
        li_attr = json.loads(parent_object_record.li_attr)
        if (li_attr != {}):
            print('4.3.1')
            (format_specification, comments) = compare_facts_with_format_specification(li_attr['attribute_values'], attribute_id,  parent_object_record.name, format_specification, comments)
           
    # Additional Format Restrictions from the Meta Data
    # ------   The next SECTION WORKS WELL it's however TEMPORARILY REMOVED, because it's TOO SLOW   ------
    # uploaded_dataset = Uploaded_dataset.objects.get(id=upload_id)
    # meta_data_facts_list = json.loads(uploaded_dataset.meta_data_facts)
    # (format_specification, comments) = compare_facts_with_format_specification(meta_data_facts_list, attribute_id, "Meta Data", format_specification, comments)

    print('4.6')
    concluding_format = {}
    concluding_format['format_specification'] = format_specification
    concluding_format['comments'] = comments
    print('4.7')
    return concluding_format


def convert_fact_values_to_the_right_format(facts):
    updated_facts = []
    for fact in facts:
        attribute = Attribute.objects.get(id=fact['attribute_id'])
        data_type = json.loads(attribute.format_specification)['fields']['column']['type']
        if data_type in ['int', 'relation']:
            fact['value'] = int(fact['value'])
        elif data_type == 'real':
            fact['value'] = float(fact['value'])
        elif data_type == 'bool':
            if fact['value'].lower() in ['true','tru', 'ture','tue','t']:
                fact['value'] = True
            else:
                fact['value'] = False
        updated_facts.append(fact)

    return updated_facts


# used in edit_model.html
# TODO: CHANGE THIS LATER TO: the object types that are most commonly used in simulations (instead of: with biggest number of objects)
def get_most_commonly_used_object_types():

    top_object_groups = Object.objects.all().values('object_type_id').annotate(total=Count('object_type_id')).order_by('-total')
    top_object_type_ids = top_object_groups.values_list('object_type_id', flat=True)
    top_object_types = list(Object_types.objects.filter(id__in=top_object_type_ids).values('id','name','object_type_icon'))

    result = {}
    for object_type in top_object_types:
        object_type_id = object_type.pop('id')
        result[object_type_id] = object_type

    return result



# used in edit_model.html
# TODO: CHANGE THIS LATER TO: sort execution orders after popularity
def get_available_execution_orders():
    execution_orders = Execution_order.objects.all().values('id', 'name','description')

    result = {}
    for execution_order in execution_orders:
        execution_order_id = execution_order.pop('id')
        result[execution_order_id] = execution_order

    return result



# used in edit_model.html
def get_available_relations():
    relations = list(Attribute.objects.filter(data_type='relation').values())
    relations_dict = {}
    for relation in relations:
        applicable_object_types_list = get_list_of_child_objects(relation['first_applicable_object_type'])
        relation['all_applicable_object_types'] = [object_type['id'] for object_type in applicable_object_types_list]
        relation_object_types_list = get_list_of_child_objects(relation['first_relation_object_type'])
        relation['all_relation_object_types'] = [object_type['id'] for object_type in relation_object_types_list]
        relations_dict[relation['id']] = relation
    return relations_dict


# used in simulation.py
def get_rules_pdf(execution_order_id, rule_or_parameter_id, is_rule):
    print('-----------  get_rules_pdf(' + str(execution_order_id) + ',' + str(rule_or_parameter_id) + ', ' + str(is_rule) + ')  ----------------------')
    if is_rule:
        likelihoods_df = pd.DataFrame(Likelihood_function.objects.filter(execution_order_id=execution_order_id, rule_id=rule_or_parameter_id, nb_of_tested_parameters_in_posterior__gt=0).values())
    else:
        likelihoods_df = pd.DataFrame(Likelihood_function.objects.filter(execution_order_id=execution_order_id, parameter_id=rule_or_parameter_id, nb_of_tested_parameters_in_posterior__gt=0).values())

    print('===================================================')
    print(len(likelihoods_df))
    print(str(likelihoods_df))
    print('===================================================')
    if len(likelihoods_df) > 0:
        # multiply the likelihood functions of all different simulations/evidences to get a combined posterior
        likelihoods_df = likelihoods_df.sort_values(['simulation_id','id'], ascending=[False, False])
        likelihoods_df = likelihoods_df.groupby(['simulation_id', 'object_number']).aggregate({'list_of_probabilities':'first', 'nb_of_simulations':'first', 'nb_of_sim_in_which_rule_was_used':'first', 'nb_of_tested_parameters':'first', 'nb_of_tested_parameters_in_posterior':'first'})

        posterior_probabilities = np.array([1] * 30)
        posterior_probabilities_smooth = np.array([1] * 30)
        nb_of_simulations = 0
        nb_of_sim_in_which_rule_was_used = 0
        nb_of_tested_parameters = 0
        nb_of_tested_parameters_in_posterior = 0
        
        for index, row in likelihoods_df.iterrows():
            list_of_probabilities = json.loads(row['list_of_probabilities'])
            list_of_probabilities_smooth = np.zeros(30)

            # list_of_probabilities_smooth (kernel smoothing)
            x = np.linspace(-1, 1, 59)
            sigma = 0.03 + 1/(row['nb_of_tested_parameters'] + 1)
            weights = stats.norm.pdf(x, 0, sigma)
            for position in range(30):
                list_of_probabilities_smooth[position] = np.sum(list_of_probabilities * weights[29-position:59-position])
            list_of_probabilities_smooth = list_of_probabilities_smooth * 30/ np.sum(list_of_probabilities_smooth) 

            # multiply with likelihood function
            if not np.any(np.isnan(list_of_probabilities)):
                posterior_probabilities = posterior_probabilities * list_of_probabilities                            # <- most important line
                posterior_probabilities_smooth = posterior_probabilities_smooth * list_of_probabilities_smooth       # <- most important line
                nb_of_simulations += row['nb_of_simulations']
                nb_of_sim_in_which_rule_was_used += row['nb_of_sim_in_which_rule_was_used']
                nb_of_tested_parameters += row['nb_of_tested_parameters']
                nb_of_tested_parameters_in_posterior += row['nb_of_tested_parameters_in_posterior']
        
        # re-normalisation
        if np.sum(posterior_probabilities)>0:
            posterior_probabilities = posterior_probabilities * 30/ np.sum(posterior_probabilities) 
            posterior_probabilities_smooth = posterior_probabilities_smooth * 30/ np.sum(posterior_probabilities_smooth) 
        else: 
            posterior_probabilities = np.array([1] * 30)
        histogram = (posterior_probabilities.tolist(), np.linspace(0,1,31).tolist())
        histogram_smooth = (posterior_probabilities_smooth.tolist(), np.linspace(0,1,31).tolist())

        x_values = np.linspace(0,0.966666666666667,30) + 1/60
        mean = np.average(x_values, weights=posterior_probabilities)
        standard_dev = np.sqrt(np.average((x_values - mean)**2, weights=posterior_probabilities))
        return histogram, mean, standard_dev, nb_of_simulations, nb_of_sim_in_which_rule_was_used, nb_of_tested_parameters, nb_of_tested_parameters_in_posterior, histogram_smooth
            
    else:
        return None, None, None, 0, 0, 0, 0,None


def get_single_pdf(simulation_id, execution_order_id, object_number, rule_or_parameter_id, is_rule, smooth):
    print('----  get_single_pdf  ----')
    if is_rule:
        likelihood_function = Likelihood_function.objects.filter(simulation_id=simulation_id, execution_order_id=execution_order_id, object_number=object_number, rule_id=rule_or_parameter_id).order_by('-id').first()
    else:
        likelihood_function = Likelihood_function.objects.filter(simulation_id=simulation_id, execution_order_id=execution_order_id, object_number=object_number, parameter_id=rule_or_parameter_id).order_by('-id').first()

    if likelihood_function is not None:

        nb_of_tested_parameters_in_posterior = likelihood_function.nb_of_tested_parameters_in_posterior
        list_of_probabilities = json.loads(likelihood_function.list_of_probabilities)

        # list_of_probabilities_smooth (kernel smoothing)
        if smooth:
            list_of_probabilities_smooth = np.zeros(30)
            x = np.linspace(-1, 1, 59)
            sigma = 0.03 + 1/(row['nb_of_tested_parameters'] + 1)
            weights = stats.norm.pdf(x, 0, sigma)
            for position in range(30):
                list_of_probabilities_smooth[position] = np.sum(list_of_probabilities * weights[29-position:59-position])
            list_of_probabilities_smooth = list_of_probabilities_smooth * 30/ np.sum(list_of_probabilities_smooth) 
            histogram = (posterior_probabilities_smooth.tolist(), np.linspace(0,1,31).tolist())
        else:
            histogram = (list(list_of_probabilities), list(np.linspace(0,1,31)))

        x_values = np.linspace(0,0.966666666666667,30) + 1/60
        mean = np.average(x_values, weights=list_of_probabilities)
        standard_dev = np.sqrt(np.average((x_values - mean)**2, weights=list_of_probabilities))

        nb_of_sim_in_which_rule_was_used = likelihood_function.nb_of_sim_in_which_rule_was_used
        if (nb_of_sim_in_which_rule_was_used == 0):
            message = 'Initial distribution: uniform'
        elif (nb_of_sim_in_which_rule_was_used < 200):
            message = 'This rule was triggered in only ' + str(nb_of_sim_in_which_rule_was_used) + ' of the ' + str(likelihood_function.nb_of_simulations) + ' simulations.'
        else:
            message = ''

        return histogram, mean, standard_dev, nb_of_sim_in_which_rule_was_used, message

    else:
        return None, None, None, 0, None


    








