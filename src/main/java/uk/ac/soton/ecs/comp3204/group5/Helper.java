package uk.ac.soton.ecs.comp3204.group5;

import org.openimaj.data.dataset.*;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    public static LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> getFeatures(FImage image) {
        ArrayList<LocalFeatureImpl<SpatialLocation, FloatFV>> features = new ArrayList<>();
        RectangleSampler sampler = new RectangleSampler(image.normalise(),4,4,8,8);
        List<Rectangle> rectangles = sampler.allRectangles();
        for(Rectangle r : rectangles) {
            FImage rImg = image.normalise().extractROI(r);
            float[] rVector = rImg.getFloatPixelVector();
            //keypoints.add(new FloatKeypoint(r.x, r.y, 0, 1, rVector));
            SpatialLocation rLoc = new SpatialLocation(r.x, r.y);
            features.add(new LocalFeatureImpl<>(rLoc, new FloatFV(rVector)));
        }
        return new MemoryLocalFeatureList<>(features);
    }
    public static GroupedDataset<String, ListBackedDataset<Record>, Record> convertToGroupedDataset(VFSGroupDataset<FImage> originalDataset) {
        GroupedDataset<String, ListBackedDataset<Record>, Record> recordDataset = new MapBackedDataset<>();
        //iterate through each class in original dataset, create Record objects from images in each class
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : originalDataset.entrySet()) {
            VFSListDataset<FImage> imageList = entry.getValue();
            ListBackedDataset<Record> recordList = new ListBackedDataset<>();
            for (int i = 0; i < imageList.size(); i++) {
                FImage image = imageList.get(i);
                recordList.add(new Record(image, i+"", entry.getKey()));
            }
            recordDataset.put(entry.getKey(), recordList);
        }
        return recordDataset;
    }
}
