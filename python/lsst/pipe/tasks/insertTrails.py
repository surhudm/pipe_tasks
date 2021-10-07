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
Inject fake trailed-sources into calexps

(Adapted version of insertFakes.py + processCcdWithFakes.py)
"""
import galsim

import lsst.log
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.exceptions import LogicError
from lsst.geom import SpherePoint, radians

__all__ = ["InsertFakeTrailsConfig", "InsertFakeTrailsTask"]


class InsertFakeTrailsConnections(PipelineTaskConnections,
                                  dimensions=("instrument", "visit", "detector", "skymap")):

    exposure = cT.Input(
        doc="The image into which fake trails are to be added.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("visit", "detector", "instrument")
    )

    fakeCat = cT.Input(
        doc="Catalog of fake trailed-sources to be added",
        name="fake_trails_catalog_inst",
        storageClass="DataFrame",
        dimensions=("instrument",)
    )

    imageWithFakes = cT.Output(
        doc="Image with fake trails added.",
        name="fakes_calexp",
        storageClass="ExposureF",
        dimensions=("visit", "detector", "instrument")
    )


class InsertFakeTrailsConfig(PipelineTaskConfig,
                             pipelineConnections=InsertFakeTrailsConnections):
    """Config class for inserting fake trailed-sources
    """

    trimBuffer = pexConfig.Field(
        doc="Size of the pixel buffer surrounding the image. Only those fake sources with a centroid"
        "falling within the image+buffer region will be considered for fake source injection.",
        dtype=int,
        default=100,
    )

    calibFluxRadius = pexConfig.Field(
        doc="Aperture radius (in pixels) that was used to define the calibration for this image+catalog. "
        "This will be used to produce the correct instrumental fluxes within the radius. "
        "This value should match that of the field defined in slot_CalibFlux_instFlux.",
        dtype=float,
        default=12.0,
    )

    ra_col = pexConfig.Field(
        doc="Source catalog column name for RA (in radians).",
        dtype=str,
        default="ra",
    )

    dec_col = pexConfig.Field(
        doc="Source catalog column name for dec (in radians).",
        dtype=str,
        default="dec",
    )

    ra_vel_col = pexConfig.Field(
        doc="Source catalog column name for RA velocity (in radians/sec)",
        dtype=str,
        default="ra_vel",
    )

    dec_vel_col = pexConfig.Field(
        doc="Source catalog column name for DEC velocity (in radians/sec)",
        dtype=str,
        default="dec_vel",
    )

    length_col = pexConfig.Field(
        doc="Source catalog column name for trail length (in pixels).",
        dtype=str,
        default="trail_length",
    )

    angle_col = pexConfig.Field(
        doc="Source catalog column name for trail angle (in radians).",
        dtype=str,
        default="trail_angle",
    )


class InsertFakeTrailsTask(PipelineTask):
    """Insert fake trails into images.

    Inject fake trailed-sources into a given image.
    """

    ConfigClass = InsertFakeTrailsConfig
    _DefaultName = "insertFakeTrails"

    def run(self, fakeCat, exposure, wcs=None, photoCalib=None):
        """Add fake trailed-sources to an image.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
            The catalog of fake trails to be added
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
            The exposure into which the fake trails should be added
        wcs : `lsst.afw.geom.SkyWcs`
            WCS to use when adding the fake trails
        photoCalib : `lsst.afw.image.photoCalib`
            Photometric calibration to be used to calibrate fake trails

        Returns
        -------
        resultStruct : `lsst.pipe.base.struct.Struct`
            contains : image : `lsst.afw.image.exposure.exposure.ExposureF`
            The resulting image with fake trails injected.
        """

        # Save original wcs and photocalib
        origWcs = exposure.getWcs()
        origPhotoCalib = exposure.getPhotoCalib()

        # Override wcs and photocalib
        if wcs is not None:
            exposure.setWcs(wcs)
        if photoCalib is not None:
            exposure.setPhotoCalib(photoCalib)

        # Make sure objects are in exposure
        fakeCat = self.addPixCoords(fakeCat, exposure)
        fakeCat = self.trimFakeCat(fakeCat, exposure)

        # Add fake trail to image
        if len(fakeCat) > 0:
            generator = self._generateTrailGSObjectsFromCatalog(exposure, fakeCat)
            self.addFakeTrailedSources(exposure, generator, calibFluxRadius=self.config.calibFluxRadius)
        elif len(fakeCat) == 0:
            lsst.log.info("No sources in image")
        else:
            raise RuntimeError("No fake trails found in dataRef.")

        # Restore original exposure WCS and photoCalib
        exposure.setWcs(origWcs)
        exposure.setPhotoCalib(origPhotoCalib)

        resultStruct = pipeBase.Struct(imageWithFakes=exposure)
        return resultStruct

    def addPixCoords(self, fakeCat, exposure):

        """Add pixel coordinates to the catalog of fakes.

        Reimplemented from lsst.pipe.tasks.insertFakes.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
                    The image into which the fake sources should be added

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
        """

        wcs = exposure.getWcs()
        ras = fakeCat['ra'].values
        decs = fakeCat['dec'].values
        xs, ys = wcs.skyToPixelArray(ras, decs)
        fakeCat["centroid_x"] = xs
        fakeCat["centroid_y"] = ys

        return fakeCat

    def trimFakeCat(self, fakeCat, exposure):
        """Trim the fake cat to about the size of the input image.

        `fakeCat` must be processed with addPixCoords before using this method.
        Reimplemented from lsst.pipe.tasks.insertFakes.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
                    The image into which the fake sources should be added

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
                    The original fakeCat trimmed to the area of the image
        """

        bbox = exposure.getBBox().dilatedBy(self.config.trimBuffer)
        xs = fakeCat["centroid_x"].values
        ys = fakeCat["centroid_y"].values

        isContained = xs >= bbox.minX
        isContained &= xs <= bbox.maxX
        isContained &= ys >= bbox.minY
        isContained &= ys <= bbox.maxY

        return fakeCat[isContained]

    def addFakeTrailedSources(self, exposure, trails, calibFluxRadius=12.0):
        """Add a fake trailed-source to an image

        (Adapted from `_add_fake_sources` in insertFakes.py)

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
            The exposure into which the fake sources should be added
        trails : `typing.Iterator` [`tuple` [`lsst.geom.SpherePoint`, `galsim.GSObject`, `float`]]
            An iterator of tuples containing the location, surface brightness
            profile, and total flux in the trail.
        calibFluxRadius : `float`, optional
            Aperture radius (in pixels) used to define the calibration for this
            expsure+catalog.

        See also
        --------
        lsst.pipe.tasks.insertFakes._add_fake_sources
        """

        # Setup and get the FAKE mask plane
        exposure.mask.addMaskPlane("FAKE")
        bitmask = exposure.mask.getPlaneBitMask("FAKE")

        # Get exposure properties
        wcs = exposure.getWcs()
        psf = exposure.getPsf()
        bbox = exposure.getBBox()
        fullBounds = galsim.BoundsI(bbox.minX, bbox.maxX, bbox.minY, bbox.maxY)
        gsImg = galsim.Image(exposure.image.array, bounds=fullBounds)

        for spt, trailObj, flux in trails:
            # Map center from sky to pixel coords
            center = wcs.skyToPixel(spt)
            posd = galsim.PositionD(center.x, center.y)
            posi = galsim.PositionI(center.x//1, center.y//1)

            # Get WCS for galsim
            mat = wcs.linearizePixelToSky(spt, geom.arcseconds).getMatrix()
            gsWCS = galsim.JacobianWCS(mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1])

            # Compute normalized PSF image at center of trail
            psfArr = psf.computeKernelImage(center).array
            apCorr = psf.computeApertureFlux(calibFluxRadius)
            psfArr /= apCorr

            # Make galsim PSF image
            gsPSF = galsim.InterpolatedImage(galsim.Image(psfArr), wcs=gsWCS)

            # Convolve a 'trail' galsim object with the PSF
            conv = galsim.Convolve(trailObj, gsPSF)
            conv = conv.withFlux(flux)  # Set total flux in resulting image

            # Generate postage stamp of trail
            stampSize = conv.getGoodImageSize(gsWCS.minLinearScale())
            subBounds = galsim.BoundsI(posi).withBorder(stampSize//2)
            subBounds &= fullBounds  # Make sure we're inside the exposure

            if subBounds.area() <= 0:
                lsst.log.warn("subBounds area is <0 for trail at x={0}, y={1}".format(center.x, center.y))
                return

            subImg = gsImg[subBounds]

            # Add postage stamp to exposure
            offset = posd - subBounds.true_center  # Shift to galsim coordinates
            conv.drawImage(
                subImg,
                add_to_image=True,
                offset=offset,
                # wcs=gsWCS,
                method='no_pixel'
            )

            # Add mask to "FAKE" mask plane
            subBox = geom.Box2I(
                geom.Point2I(subBounds.xmin, subBounds.ymin),
                geom.Point2I(subBounds.xmax, subBounds.ymax)
            )
            exposure[subBox].mask.array |= bitmask

    def _generateTrailGSObjectsFromCatalog(self, exposure, fakeCat):
        """Generate `galsim.GSObject` s from fakes catalog.

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
            The exposure for which to add fake trailed-sources
        fakeCat : `pandas.core.frame.DataFrame`
            Input catalog of fake trails
        nPerPixel : `int`, optional
            Number of per-pixel positions to generate a `galsim.DeltaFunction`

        Yields
        ------
        trailObjects : `generator`
            A generator of tuples of `lsst.geom.SpherePoint`, `galsim.GSObject`,
            and `float`.

        Notes
        -----
        Fake trails are created by making a ``line`` of `galsim.DeltaFunction`
        sources and convolving with the PSF. The number of delta functions to
        generate per image pixel is set by `config.nPerPixel`, which defaults to
        10.

        See Also
        --------
        lsst.pipe.tasks.insertFakes._generateGSObjectsFromCatalog
        """
        wcs = exposure.getWcs()
        photoCalib = exposure.getPhotoCalib()
        pixelscale = wcs.getPixelScale().asArcseconds()

        for (index, row) in fakeCat.iterrows():
            # Get center of the trail in pixel coordinates
            ra = row['ra']
            dec = row['dec']
            skyCoord = SpherePoint(ra, dec, radians)
            center = wcs.skyToPixel(skyCoord)

            # Get a flux value or skip source
            try:
                flux = photoCalib.magnitudeToInstFlux(row['mag'], center)
            except LogicError:
                continue

            # Get trail parameters
            length = row['trail_length']
            angle = row['trail_angle']

            # Make a 'trail' GSObject
            gs_length = length*pixelscale  # Transform length to arcseconds
            gs_thickness = 1e-6  # Make a 'thin' box profile
            trail = galsim.Box(gs_length, gs_thickness)

            # Rotate box through theta
            theta = galsim.Angle(angle * galsim.radians)  # Make galsim Angle
            trail = trail.rotate(theta)

            # Make sure flux is 1 (will be set after PSF convolution)
            trail = trail.withFlux(flux)

            yield skyCoord, trail, flux

    def _getMetadataName(self):
        """Disable metadata writing"""
        return None
