####################################################################
# This file is part of the Tree of Knowledge project.
# Copyright (c) 2019-2040 Benedikt Kleppmann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version - see http://www.gnu.org/licenses/.
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







    