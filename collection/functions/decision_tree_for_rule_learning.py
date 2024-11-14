####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (C) Benedikt Kleppmann - All Rights Reserved
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Benedikt Kleppmann <benedikt@kleppmann.de>, February 2021
#####################################################################

from collection.models import Learned_rule
import json
import pandas as pd
from collection.functions import query_datapoints
from patsy import (ModelDesc, Term, EvalFactor, LookupFactor, dmatrices)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.stats import norm
import numpy as np
import re
import math


# called from learn_rule.html
class Decision_Tree_for_Rule_Learning:
 


    def __init__(self, learned_rule_id):

        self.learned_rule_id = learned_rule_id
        learned_rule_record = Learned_rule.objects.get(id=learned_rule_id)







            

    # ==========================================================================================
    #  Run
    # ==========================================================================================

    def run(self):
        self.__run_linear_regression()
        self.__prepare_response_data()

        results_data = {'overall_score':self.overall_score,
                        'specified_factors': self.specified_factors}

        return results_data



    def __run_linear_regression(self):

   






    def __prepare_response_data(self):
 

        # Save results to Learned Rule Object
        learned_rule_record = Learned_rule.objects.get(id=self.learned_rule_id)
        learned_rule_record.specified_factors = json.dumps(self.specified_factors)
        learned_rule_record.sorted_factor_numbers = json.dumps(self.sorted_factor_numbers)
        learned_rule_record.overall_score = json.dumps(self.overall_score)
        learned_rule_record.save()




 
            

      

    # ==========================================================================================
    #  Getter-Functions
    # ==========================================================================================
    def get_attribute_id(self):
        return self.attribute_id







    