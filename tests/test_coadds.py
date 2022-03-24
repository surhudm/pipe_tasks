# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Unit tests for coaddition  tasks.

For full integration tests, see ci_hsc_gen3/tests/test_coadd_outputs.py
"""
import unittest

import lsst.utils.tests

from lsst.pipe.tasks.assembleCoadd import AssembleCoaddConfig, SafeClipAssembleCoaddConfig


class AssembleCoaddTestCase(lsst.utils.tests.TestCase):

    def testSafeClipConfig(self):
        # Test for DM-4797: ensure that AssembleCoaddConfig.setDefaults() is
        # run when SafeClipAssembleCoaddConfig.setDefaults() is run. This
        # simply sets the default value for badMaskPlanes.
        self.assertEqual(AssembleCoaddConfig().badMaskPlanes, SafeClipAssembleCoaddConfig().badMaskPlanes)


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    unittest.main()