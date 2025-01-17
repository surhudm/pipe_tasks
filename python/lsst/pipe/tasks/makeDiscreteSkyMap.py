#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import sys
import traceback
import lsst.sphgeom

import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import DiscreteSkyMap, BaseSkyMap
from lsst.pipe.base import ArgumentParser
from lsst.utils.timer import timeMethod


class MakeDiscreteSkyMapConfig(pexConfig.Config):
    """Config for MakeDiscreteSkyMapTask
    """
    coaddName = pexConfig.Field(
        doc="coadd name, e.g. deep, goodSeeing, chiSquared",
        dtype=str,
        default="deep",
    )
    skyMap = pexConfig.ConfigField(
        dtype=BaseSkyMap.ConfigClass,
        doc="SkyMap configuration parameters, excluding position and radius"
    )
    borderSize = pexConfig.Field(
        doc="additional border added to the bounding box of the calexps, in degrees",
        dtype=float,
        default=0.0
    )
    doAppend = pexConfig.Field(
        doc="append another tract to an existing DiscreteSkyMap on disk, if present?",
        dtype=bool,
        default=False
    )
    doWrite = pexConfig.Field(
        doc="persist the skyMap?",
        dtype=bool,
        default=True,
    )

    def setDefaults(self):
        self.skyMap.tractOverlap = 0.0


class MakeDiscreteSkyMapRunner(pipeBase.TaskRunner):
    """Run a task with all dataRefs at once, rather than one dataRef at a time.

    Call the run method of the task using two positional arguments:
    - butler: data butler
    - dataRefList: list of all dataRefs,
    """
    @staticmethod
    def getTargetList(parsedCmd):
        return [(parsedCmd.butler, parsedCmd.id.refList)]

    def __call__(self, args):
        """
        @param args     Arguments for Task.run()

        @return:
        - None if self.doReturnResults false
        - A pipe_base Struct containing these fields if self.doReturnResults true:
            - dataRef: the provided data reference
            - metadata: task metadata after execution of run
            - result: result returned by task run, or None if the task fails
        """
        butler, dataRefList = args
        task = self.TaskClass(config=self.config, log=self.log)
        result = None  # in case the task fails
        exitStatus = 0  # exit status for shell
        if self.doRaise:
            result = task.runDataRef(butler, dataRefList)
        else:
            try:
                result = task.runDataRef(butler, dataRefList)
            except Exception as e:
                task.log.fatal("Failed: %s", e)
                exitStatus = 1
                if not isinstance(e, pipeBase.TaskError):
                    traceback.print_exc(file=sys.stderr)
        for dataRef in dataRefList:
            task.writeMetadata(dataRef)

        if self.doReturnResults:
            return pipeBase.Struct(
                dataRefList=dataRefList,
                metadata=task.metadata,
                result=result,
                exitStatus=exitStatus,
            )
        else:
            return pipeBase.Struct(
                exitStatus=exitStatus,
            )


class MakeDiscreteSkyMapTask(pipeBase.CmdLineTask):
    """!Make a DiscreteSkyMap in a repository, using the bounding box of a set of calexps.

    The command-line and run signatures and config are sufficiently different from MakeSkyMapTask
    that we don't inherit from it, but it is a replacement, so we use the same config/metadata names.
    """
    ConfigClass = MakeDiscreteSkyMapConfig
    _DefaultName = "makeDiscreteSkyMap"
    RunnerClass = MakeDiscreteSkyMapRunner

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runDataRef(self, butler, dataRefList):
        """Make a skymap from the bounds of the given set of calexps using the butler.

        Parameters
        ----------
        butler : `lsst.daf.persistence.Butler`
           Gen2 data butler used to save the SkyMap
        dataRefList : iterable
           A list of Gen2 data refs of calexps used to determin the size and pointing of the SkyMap
        Returns
        -------
        struct : `lsst.pipe.base.Struct`
           The returned struct has one attribute, ``skyMap``, which holds the returned SkyMap
        """
        wcs_bbox_tuple_list = []
        oldSkyMap = None
        datasetName = self.config.coaddName + "Coadd_skyMap"
        for dataRef in dataRefList:
            if not dataRef.datasetExists("calexp"):
                self.log.warning("CalExp for %s does not exist: ignoring", dataRef.dataId)
                continue
            wcs_bbox_tuple_list.append((dataRef.get("calexp_wcs", immediate=True),
                                        dataRef.get("calexp_bbox", immediate=True)))
        if self.config.doAppend and butler.datasetExists(datasetName):
            oldSkyMap = butler.get(datasetName, immediate=True)
            if not isinstance(oldSkyMap.config, DiscreteSkyMap.ConfigClass):
                raise TypeError("Cannot append to existing non-discrete skymap")
            compareLog = []
            if not self.config.skyMap.compare(oldSkyMap.config, output=compareLog.append):
                raise ValueError("Cannot append to existing skymap - configurations differ:", *compareLog)
        result = self.run(wcs_bbox_tuple_list, oldSkyMap)
        if self.config.doWrite:
            butler.put(result.skyMap, datasetName)
        return result

    @timeMethod
    def run(self, wcs_bbox_tuple_list, oldSkyMap=None):
        """Make a SkyMap from the bounds of the given set of calexp metadata.

        Parameters
        ----------
        wcs_bbox_tuple_list : iterable
           A list of tuples with each element expected to be a (Wcs, Box2I) pair
        oldSkyMap : `lsst.skymap.DiscreteSkyMap`, option
           The SkyMap to extend if appending
        Returns
        -------
        struct : `lsst.pipe.base.Struct
           The returned struct has one attribute, ``skyMap``, which holds the returned SkyMap
        """
        self.log.info("Extracting bounding boxes of %d images", len(wcs_bbox_tuple_list))
        points = []
        for wcs, boxI in wcs_bbox_tuple_list:
            boxD = geom.Box2D(boxI)
            points.extend(wcs.pixelToSky(corner).getVector() for corner in boxD.getCorners())
        if len(points) == 0:
            raise RuntimeError("No data found from which to compute convex hull")
        self.log.info("Computing spherical convex hull")
        polygon = lsst.sphgeom.ConvexPolygon.convexHull(points)
        if polygon is None:
            raise RuntimeError(
                "Failed to compute convex hull of the vertices of all calexp bounding boxes; "
                "they may not be hemispherical."
            )
        circle = polygon.getBoundingCircle()

        skyMapConfig = DiscreteSkyMap.ConfigClass()
        if oldSkyMap:
            skyMapConfig.raList.extend(oldSkyMap.config.raList)
            skyMapConfig.decList.extend(oldSkyMap.config.decList)
            skyMapConfig.radiusList.extend(oldSkyMap.config.radiusList)
        configIntersection = {k: getattr(self.config.skyMap, k)
                              for k in self.config.skyMap.toDict()
                              if k in skyMapConfig}
        skyMapConfig.update(**configIntersection)
        circleCenter = lsst.sphgeom.LonLat(circle.getCenter())
        skyMapConfig.raList.append(circleCenter[0].asDegrees())
        skyMapConfig.decList.append(circleCenter[1].asDegrees())
        circleRadiusDeg = circle.getOpeningAngle().asDegrees()
        skyMapConfig.radiusList.append(circleRadiusDeg + self.config.borderSize)
        skyMap = DiscreteSkyMap(skyMapConfig)

        for tractInfo in skyMap:
            wcs = tractInfo.getWcs()
            posBox = geom.Box2D(tractInfo.getBBox())
            pixelPosList = (
                posBox.getMin(),
                geom.Point2D(posBox.getMaxX(), posBox.getMinY()),
                posBox.getMax(),
                geom.Point2D(posBox.getMinX(), posBox.getMaxY()),
            )
            skyPosList = [wcs.pixelToSky(pos).getPosition(geom.degrees) for pos in pixelPosList]
            posStrList = ["(%0.3f, %0.3f)" % tuple(skyPos) for skyPos in skyPosList]
            self.log.info("tract %s has corners %s (RA, Dec deg) and %s x %s patches",
                          tractInfo.getId(), ", ".join(posStrList),
                          tractInfo.getNumPatches()[0], tractInfo.getNumPatches()[1])
        return pipeBase.Struct(
            skyMap=skyMap
        )

    def _getConfigName(self):
        """Return None to disable saving config

        There's only one SkyMap per repository, so the config is redundant, and checking it means we can't
        easily overwrite or append to an existing repository.
        """
        return None

    def _getMetadataName(self):
        """Return None to disable saving metadata

        The metadata is not interesting, and by not saving it we can eliminate a dataset type.
        """
        return None

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="calexp", help="data ID, e.g. --id visit=123 ccd=1,2")
        return parser
