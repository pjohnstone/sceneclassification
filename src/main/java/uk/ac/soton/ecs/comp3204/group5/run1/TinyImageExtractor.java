package uk.ac.soton.ecs.comp3204.group5.run1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

public class TinyImageExtractor implements FeatureExtractor<DoubleFV, FImage> {

    private int size;

    public TinyImageExtractor(int size){
        this.size = size;
    }

    /**
     * Generates a zero mean, unit length feature vector image representing a tiny image
     * @param image: input image
     * @return
     */
    @Override
    public DoubleFV extractFeature(FImage image){
        // Get smallest side and crop image so both width and height are equal to smallest side
        int minSide = Math.min(image.height , image.width);
        FImage cropped = image.extractCenter(minSide, minSide);

        // Resizes image to 16 by 16 pixels (or generally size by size)
        FImage tinyImage = ResizeProcessor.resample(cropped, this.size, this.size);

        // Mean center the image
        float mean = tinyImage.sum() / (float) Math.pow(this.size, 2);
        FImage centeredTinyImage = tinyImage.subtract(mean);

        // Return normalised FV
        return new DoubleFV(centeredTinyImage.getDoublePixelVector()).normaliseFV();
    }
}
