package uk.ac.soton.ecs.comp3204;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;

import java.util.ArrayList;
import java.util.List;

public class Helper {

    public static LocalFeatureList<FloatKeypoint> createLocalFeatureList(FImage image) {
        ArrayList<FloatKeypoint> keypoints = new ArrayList<>();
        RectangleSampler sampler = new RectangleSampler(image.normalise(),4,4,8,8);
        List<Rectangle> rectangles = sampler.allRectangles();
        for(Rectangle r : rectangles) {
            FImage rImg = image.normalise().extractROI(r);
            float[] rVector = rImg.getFloatPixelVector();
            keypoints.add(new FloatKeypoint(r.x, r.y, 0, 1, rVector));
        }
        return new MemoryLocalFeatureList<>(keypoints);
    }
}
