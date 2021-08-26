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
Insert fake trailed-sources into calexps
"""

import lsst.meas.extensions.trailedSources

import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

from .insertTrails import InsertFakeTrailsTask
from lsst.afw.table import SourceTable
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.tasks.calibrate import CalibrateTask

__all__ = ["ProcessCcdWithFakeTrailsConfig", "ProcessCcdWithFakeTrailsTask"]


class ProcessCcdWithFakeTrailsConnections(PipelineTaskConnections,
                                          dimensions=("visit", "detector", "skymap")):
    exposure = cT.Input(
        doc="Exposure to add fake trails to.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector")
    )

    fakeCat = cT.Input(
        doc="Catalog of fake sources to be added",
        name="fake_trails_catalog",
        storageClass="DataFrame",
        dimensions=("skymap",)
    )

    trueSourceCat = cT.Input(
        doc="Catalog of calibration sources before injection.",
        name="src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector")
    )

    outputExposure = cT.Output(
        doc="Exposure with fake trails added.",
        name="fakes_calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector")
    )

    outputCat = cT.Output(
        doc="Source catalog produced in calibrate traks with fakes also measured.",
        name="fakes_src",
        storageClass="SourceCatalog",
        dimensions=("instrument", "visit", "detector")
    )


class ProcessCcdWithFakeTrailsConfig(PipelineTaskConfig,
                                     pipelineConnections=ProcessCcdWithFakeTrailsConnections):

    srcFieldsToCopy = pexConfig.ListField(
        dtype=str,
        default=("calib_photometry_reserved", "calib_photometry_used", "calib_astrometry_used",
                 "calib_psf_candidate", "calib_psf_used", "calib_psf_reserved"),
        doc=("Fields to copy from the `src` catalog to the output catalog "
             "for matching sources Any missing fields will trigger a "
             "RuntimeError exception.")
    )

    matchRadiusPix = pexConfig.Field(
        dtype=float,
        default=3,
        doc=("Match radius for matching icSourceCat objects to sourceCat objects (pixels)"),
    )

    calibrate = pexConfig.ConfigurableField(target=CalibrateTask,
                                            doc="Calibration task to use.")

    insertFakeTrails = pexConfig.ConfigurableField(target=InsertFakeTrailsTask,
                                                   doc="Fake trails injection task.")

    def setDefaults(self):
        super().setDefaults()
        self.calibrate.measurement.plugins["base_PixelFlags"].masksFpAnywhere.append("FAKE")
        self.calibrate.measurement.plugins["base_PixelFlags"].masksFpCenter.append("FAKE")
        self.calibrate.doAstrometry = False
        self.calibrate.doWriteMatches = False
        self.calibrate.doPhotoCal = False
        self.calibrate.detection.reEstimateBackground = False
        self.calibrate.measurement.plugins.names |= ["ext_trailedSources_Naive", "ext_trailedSources_Veres"]


class ProcessCcdWithFakeTrailsTask(PipelineTask):
    """Insert fake trailed-sources into calexps.

    """

    _DefaultName = "processCcdWithFakeTrails"
    ConfigClass = ProcessCcdWithFakeTrailsConfig

    def __init__(self, schema=None, butler=None, **kwargs):
        super().__init__(**kwargs)

        # Make sure schema exists
        if schema is None:
            self.schema = SourceTable.makeMinimalSchema()

        self.makeSubtask("insertFakeTrails")
        self.makeSubtask("calibrate")

    def run(self, fakeCat, exposure, trueSourceCat, wcs=None, photoCalib=None):

        # insert fake trails into exposure
        self.insertFakeTrails.run(fakeCat, exposure, wcs, photoCalib)

        # Run detection, deblending, and measurement
        calibStruct = self.calibrate.run(exposure)
        sources = calibStruct.sourceCat

        sources = self.copyCalibrationFields(trueSourceCat, sources, self.config.srcFieldsToCopy)

        fakesStruct = pipeBase.Struct(outputExposure=exposure, outputCat=sources)
        return fakesStruct


    def copyCalibrationFields(self, calibCat, sourceCat, fieldsToCopy):
        """Match sources in calibCat and sourceCat and copy the specified fields

        Parameters
        ----------
        calibCat : `lsst.afw.table.SourceCatalog`
            Catalog from which to copy fields.
        sourceCat : `lsst.afw.table.SourceCatalog`
            Catalog to which to copy fields.
        fieldsToCopy : `lsst.pex.config.listField.List`
            Fields to copy from calibCat to SoourceCat.

        Returns
        -------
        newCat : `lsst.afw.table.SourceCatalog`
            Catalog which includes the copied fields.

        The fields copied are those specified by `fieldsToCopy` that actually exist
        in the schema of `calibCat`.

        This version was based on and adapted from the one in calibrateTask.
        """

        # Make a new SourceCatalog with the data from sourceCat so that we can add the new columns to it
        sourceSchemaMapper = afwTable.SchemaMapper(sourceCat.schema)
        sourceSchemaMapper.addMinimalSchema(sourceCat.schema, True)

        calibSchemaMapper = afwTable.SchemaMapper(calibCat.schema, sourceCat.schema)

        # Add the desired columns from the option fieldsToCopy
        missingFieldNames = []
        for fieldName in fieldsToCopy:
            if fieldName in calibCat.schema:
                schemaItem = calibCat.schema.find(fieldName)
                calibSchemaMapper.editOutputSchema().addField(schemaItem.getField())
                schema = calibSchemaMapper.editOutputSchema()
                calibSchemaMapper.addMapping(schemaItem.getKey(), schema.find(fieldName).getField())
            else:
                missingFieldNames.append(fieldName)
        if missingFieldNames:
            raise RuntimeError(f"calibCat is missing fields {missingFieldNames} specified in "
                               "fieldsToCopy")

        if "calib_detected" not in calibSchemaMapper.getOutputSchema():
            self.calibSourceKey = calibSchemaMapper.addOutputField(afwTable.Field["Flag"]("calib_detected",
                                                                   "Source was detected as an icSource"))
        else:
            self.calibSourceKey = None

        schema = calibSchemaMapper.getOutputSchema()
        newCat = afwTable.SourceCatalog(schema)
        newCat.reserve(len(sourceCat))
        newCat.extend(sourceCat, sourceSchemaMapper)

        # Set the aliases so it doesn't complain.
        for k, v in sourceCat.schema.getAliasMap().items():
            newCat.schema.getAliasMap().set(k, v)

        select = newCat["deblend_nChild"] == 0
        matches = afwTable.matchXy(newCat[select], calibCat, self.config.matchRadiusPix)
        # Check that no sourceCat sources are listed twice (we already know
        # that each match has a unique calibCat source ID, due to using
        # that ID as the key in bestMatches)
        numMatches = len(matches)
        numUniqueSources = len(set(m[1].getId() for m in matches))
        if numUniqueSources != numMatches:
            self.log.warning("%d calibCat sources matched only %d sourceCat sources", numMatches,
                             numUniqueSources)

        self.log.info("Copying flags from calibCat to sourceCat for %s sources", numMatches)

        # For each match: set the calibSourceKey flag and copy the desired
        # fields
        for src, calibSrc, d in matches:
            if self.calibSourceKey:
                src.setFlag(self.calibSourceKey, True)
            # src.assign copies the footprint from calibSrc, which we don't want
            # (DM-407)
            # so set calibSrc's footprint to src's footprint before src.assign,
            # then restore it
            calibSrcFootprint = calibSrc.getFootprint()
            try:
                calibSrc.setFootprint(src.getFootprint())
                src.assign(calibSrc, calibSchemaMapper)
            finally:
                calibSrc.setFootprint(calibSrcFootprint)

        return newCat
