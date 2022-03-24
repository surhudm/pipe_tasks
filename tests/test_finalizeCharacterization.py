# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test FinalizeCharacterizationTask.
"""
import unittest
import numpy as np
import pandas as pd

import lsst.utils.tests
import lsst.daf.butler
import lsst.pipe.base as pipeBase

from lsst.pipe.tasks.finalizeCharacterization import (FinalizeCharacterizationConfig,
                                                      FinalizeCharacterizationTask)


class MockDataFrameReference(lsst.daf.butler.DeferredDatasetHandle):
    """Very simple object that looks like a Gen3 data reference to
    a dataframe.
    """
    def __init__(self, df):
        self.df = df

    def get(self, parameters={}, **kwargs):
        """Retrieve the specified dataset using the API of the Gen3 Butler.

        Parameters
        ----------
        parameters : `dict`, optional
            Parameter dictionary.  Supported key is ``columns``.

        Returns
        -------
        dataframe : `pandas.DataFrame`
            dataframe, cut to the specified columns.
        """
        if 'columns' in parameters:
            _columns = parameters['columns']

            return self.df[_columns]
        else:
            return self.df.copy()


class TestFinalizeCharacterizationTask(FinalizeCharacterizationTask):
    """A derived class which skips the initialization routines.
    """
    __test__ = False  # Stop Pytest from trying to parse as a TestCase

    def __init__(self, **kwargs):
        pipeBase.PipelineTask.__init__(self, **kwargs)

        self.makeSubtask('reserve_selection')


class FinalizeCharacterizationTestCase(lsst.utils.tests.TestCase):
    """Tests of some functionality of FinalizeCharacterizationTask.

    Full testing comes from integration tests such as ci_hsc and ci_imsim.

    These tests bypass the middleware used for accessing data and
    managing Task execution.
    """
    def setUp(self):
        config = FinalizeCharacterizationConfig()

        self.finalizeCharacterizationTask = TestFinalizeCharacterizationTask(
            config=config,
        )

        self.isolated_star_cat_dict, self.isolated_star_source_dict = self._make_isocats()

    def _make_isocats(self):
        """Make test isolated star catalogs.

        Returns
        -------
        isolated_star_cat_dict : `dict`
            Per-"tract" dict of isolated star catalogs.
        isolate_star_source_dict : `dict`
            Per-"tract" dict of isolated source catalogs.
        """
        dtype_cat = [('isolated_star_id', 'i8'),
                     ('ra', 'f8'),
                     ('decl', 'f8'),
                     ('primary_band', 'U2'),
                     ('source_cat_index', 'i4'),
                     ('nsource', 'i4'),
                     ('source_cat_index_i', 'i4'),
                     ('nsource_i', 'i4'),
                     ('source_cat_index_r', 'i4'),
                     ('nsource_r', 'i4'),
                     ('source_cat_index_z', 'i4'),
                     ('nsource_z', 'i4')]

        dtype_source = [('sourceId', 'i8'),
                        ('obj_index', 'i4')]

        isolated_star_cat_dict = {}
        isolated_star_source_dict = {}

        np.random.seed(12345)

        # There are 90 stars in both r, i.  10 individually in each.
        nstar = 110
        nsource_per_band_per_star = 2
        self.nstar_total = nstar
        self.nstar_per_band = nstar - 10

        # This is a brute-force assembly of a star catalog and matched sources.
        for tract in [0, 1, 2]:
            ra = np.random.uniform(low=tract, high=tract + 1.0, size=nstar)
            dec = np.random.uniform(low=0.0, high=1.0, size=nstar)

            cat = np.zeros(nstar, dtype=dtype_cat)
            cat['isolated_star_id'] = tract*nstar + np.arange(nstar)
            cat['ra'] = ra
            cat['decl'] = dec
            if tract < 2:
                cat['primary_band'][0: 100] = 'i'
                cat['primary_band'][100:] = 'r'
            else:
                # Tract 2 only has z band.
                cat['primary_band'][:] = 'z'

            source_cats = []
            counter = 0
            for i in range(cat.size):
                cat['source_cat_index'][i] = counter
                if tract < 2:
                    if i < 90:
                        cat['nsource'][i] = 2*nsource_per_band_per_star
                        bands = ['r', 'i']
                    else:
                        cat['nsource'][i] = nsource_per_band_per_star
                        if i < 100:
                            bands = ['i']
                        else:
                            bands = ['r']
                else:
                    cat['nsource'][i] = nsource_per_band_per_star
                    bands = ['z']

                for band in bands:
                    cat[f'source_cat_index_{band}'][i] = counter
                    cat[f'nsource_{band}'][i] = nsource_per_band_per_star
                    source_cat = np.zeros(nsource_per_band_per_star, dtype=dtype_source)
                    source_cat['sourceId'] = np.arange(
                        tract*nstar + counter,
                        tract*nstar + counter + nsource_per_band_per_star
                    )
                    source_cat['obj_index'] = i

                    source_cats.append(source_cat)

                    counter += nsource_per_band_per_star

            source_cat = np.concatenate(source_cats)

            isolated_star_cat_dict[tract] = MockDataFrameReference(pd.DataFrame(cat))
            isolated_star_source_dict[tract] = MockDataFrameReference(pd.DataFrame(source_cat))

        return isolated_star_cat_dict, isolated_star_source_dict

    def test_concat_isolated_star_cats(self):
        """Test concatenation and reservation of the isolated star catalogs.
        """

        for band in ['r', 'i']:
            iso, iso_src = self.finalizeCharacterizationTask.concat_isolated_star_cats(
                band,
                self.isolated_star_cat_dict,
                self.isolated_star_source_dict
            )

            # There are two tracts, so double everything.
            self.assertEqual(len(iso), 2*self.nstar_per_band)

            reserve_fraction = self.finalizeCharacterizationTask.config.reserve_selection.reserve_fraction
            self.assertEqual(np.sum(iso['reserved']),
                             int(reserve_fraction*len(iso)))

            # 2 tracts, 4 observations per tract per star, minus 2*10 not in the given band.
            self.assertEqual(len(iso_src), 2*(4*len(iso)//2 - 20))

            # Check that every star is properly matched to the sources.
            for i in range(len(iso)):
                np.testing.assert_array_equal(
                    iso_src['obj_index'][iso[f'source_cat_index_{band}'][i]:
                                         iso[f'source_cat_index_{band}'][i] + iso[f'nsource_{band}'][i]],
                    i
                )

            # Check that every reserved star is marked as a reserved source.
            res_star, = np.where(iso['reserved'])
            for i in res_star:
                np.testing.assert_array_equal(
                    iso_src['reserved'][iso[f'source_cat_index_{band}'][i]:
                                        iso[f'source_cat_index_{band}'][i] + iso[f'nsource_{band}'][i]],
                    True
                )

            # Check that every reserved source is marked as a reserved star.
            res_src, = np.where(iso_src['reserved'])
            np.testing.assert_array_equal(
                iso['reserved'][iso_src['obj_index'][res_src]],
                True
            )

    def test_concat_isolate_star_cats_no_sources(self):
        """Test concatenation when there are no sources in a tract."""
        iso, iso_src = self.finalizeCharacterizationTask.concat_isolated_star_cats(
            'z',
            self.isolated_star_cat_dict,
            self.isolated_star_source_dict
        )

        self.assertGreater(len(iso), 0)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()