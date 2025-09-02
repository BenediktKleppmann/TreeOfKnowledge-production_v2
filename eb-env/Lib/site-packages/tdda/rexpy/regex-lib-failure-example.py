# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import re
import regex
import unittest


class TestRegex(unittest.TestCase):
    s = 'FundaciÃ³ Rego'
    R = '^[^\W0-9_]+ Rego$'

    self.assertIsNotNone(re.match(R, s, re.U))

    self.assertIsNotNone(regex.match(R, s, regex.U))


if __name__ == '__main__':
    unittest.main()


