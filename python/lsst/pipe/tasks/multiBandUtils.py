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
import lsst.afw.table as afwTable

from lsst.coadd.utils import ExistingCoaddDataIdContainer
from lsst.coadd.utils.getExposureId import getExposureId
from lsst.pipe.base import TaskRunner, ArgumentParser
from lsst.pex.config import Config, RangeField
from lsst.obs.base import ExposureIdInfo


class MergeSourcesRunner(TaskRunner):
    """Task runner for `MergeDetectionTask` `MergeMeasurementTask`

    Required because the run method requires a list of
    dataRefs rather than a single dataRef.
    """
    def makeTask(self, parsedCmd=None, args=None):
        """Provide a butler to the Task constructor.

        Parameters
        ----------
        parsedCmd:
            The parsed command
        args: tuple
            Tuple of a list of data references and kwargs (un-used)

        Raises
        ------
        RuntimeError
            Thrown if both `parsedCmd` & `args` are `None`
        """
        if parsedCmd is not None:
            butler = parsedCmd.butler
        elif args is not None:
            dataRefList, kwargs = args
            butler = dataRefList[0].getButler()
        else:
            raise RuntimeError("Neither parsedCmd or args specified")
        return self.TaskClass(config=self.config, log=self.log, butler=butler)

    @staticmethod
    def buildRefDict(parsedCmd):
        """Build a hierarchical dictionary of patch references

        Parameters
        ----------
        parsedCmd:
            The parsed command

        Returns
        -------
        refDict: dict
            A reference dictionary of the form {patch: {tract: {filter: dataRef}}}

        Raises
        ------
        RuntimeError
            Thrown when multiple references are provided for the same
            combination of tract, patch and filter
        """
        refDict = {}  # Will index this as refDict[tract][patch][filter] = ref
        for ref in parsedCmd.id.refList:
            tract = ref.dataId["tract"]
            patch = ref.dataId["patch"]
            filter = ref.dataId["filter"]
            if tract not in refDict:
                refDict[tract] = {}
            if patch not in refDict[tract]:
                refDict[tract][patch] = {}
            if filter in refDict[tract][patch]:
                raise RuntimeError("Multiple versions of %s" % (ref.dataId,))
            refDict[tract][patch][filter] = ref
        return refDict

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Provide a list of patch references for each patch, tract, filter combo.

        Parameters
        ----------
        parsedCmd:
            The parsed command
        kwargs:
            Keyword arguments passed to the task

        Returns
        -------
        targetList: list
            List of tuples, where each tuple is a (dataRef, kwargs) pair.
        """
        refDict = MergeSourcesRunner.buildRefDict(parsedCmd)
        return [(list(p.values()), kwargs) for t in refDict.values() for p in t.values()]


def _makeGetSchemaCatalogs(datasetSuffix):
    """Construct a getSchemaCatalogs instance method

    These are identical for most of the classes here, so we'll consolidate
    the code.

    datasetSuffix:  Suffix of dataset name, e.g., "src" for "deepCoadd_src"
    """

    def getSchemaCatalogs(self):
        """Return a dict of empty catalogs for each catalog dataset produced by this task."""
        src = afwTable.SourceCatalog(self.schema)
        if hasattr(self, "algMetadata"):
            src.getTable().setMetadata(self.algMetadata)
        return {self.config.coaddName + "Coadd_" + datasetSuffix: src}
    return getSchemaCatalogs


def makeMergeArgumentParser(name, dataset):
    """!
    @brief Create a suitable ArgumentParser.

    We will use the ArgumentParser to get a provide a list of data
    references for patches; the RunnerClass will sort them into lists
    of data references for the same patch
    """
    parser = ArgumentParser(name)
    parser.add_id_argument("--id", "deepCoadd_" + dataset,
                           ContainerClass=ExistingCoaddDataIdContainer,
                           help="data ID, e.g. --id tract=12345 patch=1,2 filter=g^r^i")
    return parser


def getInputSchema(task, butler=None, schema=None):
    """!
    @brief Obtain the input schema either directly or froma  butler reference.

    @param[in]  butler   butler reference to obtain the input schema from
    @param[in]  schema   the input schema
    """
    if schema is None:
        assert butler is not None, "Neither butler nor schema specified"
        schema = butler.get(task.config.coaddName + "Coadd_" + task.inputDataset + "_schema",
                            immediate=True).schema
    return schema


def readCatalog(task, patchRef):
    """!
    @brief Read input catalog.

    We read the input dataset provided by the 'inputDataset'
    class variable.

    @param[in]  patchRef   data reference for patch
    @return tuple consisting of the band name and the catalog
    """
    band = patchRef.get(task.config.coaddName + "Coadd_filterLabel", immediate=True).bandLabel
    catalog = patchRef.get(task.config.coaddName + "Coadd_" + task.inputDataset, immediate=True)
    task.log.info("Read %d sources for band %s: %s", len(catalog), band, patchRef.dataId)
    return band, catalog


class CullPeaksConfig(Config):
    """!
    @anchor CullPeaksConfig_

    @brief Configuration for culling garbage peaks after merging footprints.

    Peaks may also be culled after detection or during deblending; this configuration object
    only deals with culling after merging Footprints.

    These cuts are based on three quantities:
     - nBands: the number of bands in which the peak was detected
     - peakRank: the position of the peak within its family, sorted from brightest to faintest.
     - peakRankNormalized: the peak rank divided by the total number of peaks in the family.

    The formula that identifie peaks to cull is:

      nBands < nBandsSufficient
        AND (rank >= rankSufficient)
        AND (rank >= rankConsider OR rank >= rankNormalizedConsider)

    To disable peak culling, simply set nBandsSufficient=1.
    """

    nBandsSufficient = RangeField(dtype=int, default=2, min=1,
                                  doc="Always keep peaks detected in this many bands")
    rankSufficient = RangeField(dtype=int, default=20, min=1,
                                doc="Always keep this many peaks in each family")
    rankConsidered = RangeField(dtype=int, default=30, min=1,
                                doc=("Keep peaks with less than this rank that also match the "
                                     "rankNormalizedConsidered condition."))
    rankNormalizedConsidered = RangeField(dtype=float, default=0.7, min=0.0,
                                          doc=("Keep peaks with less than this normalized rank that"
                                               " also match the rankConsidered condition."))


def _makeMakeIdFactory(datasetName, includeBand=True):
    """Construct a makeIdFactory instance method

    These are identical for all the classes here, so this consolidates
    the code.

    datasetName:  Dataset name without the coadd name prefix, e.g., "CoaddId" for "deepCoaddId"
    """

    def makeIdFactory(self, dataRef):
        """Return an IdFactory for setting the detection identifiers

        The actual parameters used in the IdFactory are provided by
        the butler (through the provided data reference.
        """
        try:
            expId = getExposureId(dataRef, coaddName=self.config.coaddName, includeBand=includeBand,
                                  log=self.log)
        except Exception:
            expId = int(dataRef.get(self.config.coaddName + datasetName))
        info = ExposureIdInfo(expId, dataRef.get(self.config.coaddName + datasetName + "_bits"))
        return info.makeSourceIdFactory()
    return makeIdFactory
