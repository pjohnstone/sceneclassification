package uk.ac.soton.ecs.comp3204.group5.run1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.algorithm.MeanCenter;
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

        // Resize image to 16 by 16 pixels (or generally size by size) and then normalise
        FImage tinyImage = ResizeProcessor.resample(cropped, this.size, this.size).normalise();

        // Mean center the image
        MeanCenter meanCenter = new MeanCenter();
        meanCenter.processImage(tinyImage);

        // Return normalised FV
        return new DoubleFV(tinyImage.getDoublePixelVector());
    }
}
