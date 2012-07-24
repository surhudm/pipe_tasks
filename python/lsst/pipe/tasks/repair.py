# 
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011 LSST Corporation.
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
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import math
import lsst.pex.config as pexConfig
import lsst.afw.cameraGeom  as cameraGeom
import lsst.afw.display.ds9 as ds9
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg
import lsst.meas.algorithms.crosstalk as maCrosstalk
import lsst.pipe.base as pipeBase

import lsstDebug

class RepairConfig(pexConfig.Config):
    doCrosstalk = pexConfig.Field(
        dtype = bool,
        doc = "Correct for crosstalk",
        default = True,
    )
    crosstalkCoeffs = pexConfig.ConfigField(
        dtype = maCrosstalk.CrosstalkCoeffsConfig,
        doc = "Crosstalk coefficients",
    )

    crosstalkMaskPlane = pexConfig.Field(
        dtype = str,
        doc = "Name of Mask plane for crosstalk corrected pixels",
        default = "CROSSTALK",
    )

    minPixelToMask = pexConfig.Field(
        dtype = float,
        doc = "Minimum pixel value (in electrons) to cause crosstalkMaskPlane bit to be set",
        default = 45000,        
    )
    
    doLinearize = pexConfig.Field(
        dtype = bool,
        doc = "Correct for nonlinearity of the detector's response (ignored if coefficients are 0.0)",
        default = True,
    )
    linearizationThreshold = pexConfig.Field(
        dtype = float,
        doc = "Minimum pixel value (in electrons) to apply linearity corrections",
        default = 0.0,        
    )
    linearizationCoefficient = pexConfig.Field(
        dtype = float,
        doc = "Linearity correction coefficient",
        default = 0.0,        
    )

    doInterpolate = pexConfig.Field(
        dtype = bool,
        doc = "Interpolate over defects? (ignored unless you provide a list of defects)",
        default = True,
    )

    doCosmicRay = pexConfig.Field(
        dtype = bool,
        doc = "Find and mask out cosmic rays?",
        default = True,
    )
    cosmicray = pexConfig.ConfigField(
        dtype = measAlg.FindCosmicRaysConfig,
        doc = "Options for finding and masking cosmic rays",
    )

class RepairTask(pipeBase.Task):
    """Conversion notes:
    
    Display code should be updated once we settle on a standard way of controlling what is displayed.
    """
    ConfigClass = RepairConfig

    @pipeBase.timeMethod
    def run(self, exposure, defects=None, keepCRs=None, fixCrosstalk=None, linearize=None):
        """Repair exposure's instrumental problems

        @param[in,out] exposure Exposure to process
        @param defects Defect list
        @param keepCRs  Don't interpolate over the CR pixels (defer to pex_config if None)
        """
        assert exposure, "No exposure provided"
        psf = exposure.getPsf()
        assert psf, "No PSF provided"

        if fixCrosstalk is None:
            fixCrosstalk = self.config.doCrosstalk
        if linearize is None:
            linearize = self.config.doLinearize
            
        self.display('pre-crosstalk' if fixCrosstalk else 'before', exposure=exposure)

        if fixCrosstalk:
            self.crosstalk(exposure)

        if linearize:
            self.linearize(exposure)

        if defects is not None and self.config.doInterpolate:
            self.interpolate(exposure, defects)

        if self.config.doCosmicRay:
            self.cosmicRay(exposure, keepCRs=keepCRs)

        self.display('after', exposure=exposure)

    def crosstalk(self, exposure):
        coeffs = self.config.crosstalkCoeffs.getCoeffs()

        maCrosstalk.subtractXTalk(exposure.getMaskedImage(), coeffs,
                                  self.config.minPixelToMask, self.config.crosstalkMaskPlane)

    def linearize(self, exposure):
        """Correct for non-linearity

        @param exposure Exposure to process
        """
        assert exposure, "No exposure provided"

        image = exposure.getMaskedImage().getImage()

        ccd = cameraGeom.cast_Ccd(exposure.getDetector())

        for amp in ccd:
            if False:
                linear_threshold = amp.getElectronicParams().getLinearizationThreshold()
                linear_c = amp.getElectronicParams().getLinearizationCoefficient()
            else:
                linearizationCoefficient = self.config.linearizationCoefficient
                linearizationThreshold = self.config.linearizationThreshold

            if linearizationCoefficient == 0.0:     # nothing to do
                continue
            
            self.log.log(self.log.INFO,
                         "Applying linearity corrections to Ccd %s Amp %s" % (ccd.getId(), amp.getId()))

            if linearizationThreshold > 0:
                log10_thresh = math.log10(linearizationThreshold)

            ampImage = image.Factory(image, amp.getDataSec(), afwImage.LOCAL)

            width, height = ampImage.getDimensions()

            if linearizationThreshold <= 0:
                tmp = ampImage.Factory(ampImage, True)
                tmp.scaledMultiplies(linearizationCoefficient, ampImage)
                ampImage += tmp
            else:
                for y in range(height):
                    for x in range(width):
                        val = ampImage.get(x, y)
                        if val > linearizationThreshold:
                            val += val*linearizationCoefficient*(math.log10(val) - log10_thresh)
                            ampImage.set(x, y, val)
        
    def interpolate(self, exposure, defects):
        """Interpolate over defects

        @param[in,out] exposure Exposure to process
        @param defects Defect list
        """
        assert exposure, "No exposure provided"
        assert defects is not None, "No defects provided"
        psf = exposure.getPsf()
        assert psf, "No psf provided"

        mi = exposure.getMaskedImage()
        fallbackValue = afwMath.makeStatistics(mi, afwMath.MEANCLIP).getValue()
        measAlg.interpolateOverDefects(mi, psf, defects, fallbackValue)
        self.log.log(self.log.INFO, "Interpolated over %d defects." % len(defects))

    def cosmicRay(self, exposure, keepCRs=None):
        """Mask cosmic rays

        @param[in,out] exposure Exposure to process
        @param keepCRs  Don't interpolate over the CR pixels (defer to pex_config if None)
        """
        import lsstDebug
        display = lsstDebug.Info(__name__).display
        displayCR = lsstDebug.Info(__name__).displayCR

        assert exposure, "No exposure provided"
        psf = exposure.getPsf()
        assert psf, "No psf provided"

        # Blow away old mask
        try:
            mask = exposure.getMaskedImage().getMask()
            crBit = mask.getMaskPlane("CR")
            mask.clearMaskPlane(crBit)
        except Exception:
            pass

        mi = exposure.getMaskedImage()
        bg = afwMath.makeStatistics(mi, afwMath.MEDIAN).getValue()

        if keepCRs is None:
            keepCRs = self.config.cosmicray.keepCRs
        try:
            crs = measAlg.findCosmicRays(mi, psf, bg, pexConfig.makePolicy(self.config.cosmicray), keepCRs)
        except Exception, e:
            if display:
                ds9.mtv(exposure, title="Failed CR")
            raise
            
        num = 0
        if crs is not None:
            mask = mi.getMask()
            crBit = mask.getPlaneBitMask("CR")
            afwDet.setMaskFromFootprintList(mask, crs, crBit)
            num = len(crs)

            if display and displayCR:
                import lsst.afw.display.utils as displayUtils

                ds9.incrDefaultFrame()
                ds9.mtv(exposure, title="Post-CR")
                
                with ds9.Buffering():
                    for cr in crs:
                        displayUtils.drawBBox(cr.getBBox(), borderWidth=0.55)

        self.log.log(self.log.INFO, "Identified %s cosmic rays." % (num,))

