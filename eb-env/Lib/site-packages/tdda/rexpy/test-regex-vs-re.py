# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import re
import regex
import unittest


NAME = 'FundaciÃ³ Rego'
PATTERN = '^[^\W0-9_]+ Rego$'

class TestRegex(unittest.TestCase):

    def testRE(self):
        self.assertIsNotNone(re.match(PATTERN, NAME, re.U))

    def testRegex(self):
        self.assertIsNotNone(regex.match(PATTERN, NAME, regex.U))


if __name__ == '__main__':
    unittest.main()


