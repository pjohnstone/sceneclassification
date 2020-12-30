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
    //probably deperecated class, does the same thing as the other one but for FloatKeypoint objects
    public static LocalFeatureList<FloatKeypoint> createLocalFeatureList(FImage image) {
        ArrayList<FloatKeypoint> keypoints = new ArrayList<>();
        RectangleSampler sampler = new RectangleSampler(image.normalise(),4,4,8,8);
        List<Rectangle> rectangles = sampler.allRectangles();
        for(Rectangle r : rectangles) {
            FImage rImg = image.extractROI(r).normalise();
            float[] rVector = rImg.getFloatPixelVector();
            keypoints.add(new FloatKeypoint(r.x, r.y, 0, 1, rVector));
        }
        return new MemoryLocalFeatureList<>(keypoints);
    }

    /**
     * Creates a local feature list of densely sampled pixel patches flattened into a vector
     * @param image - the image to be sampled
     * @return - a local feature list
     */
    public static LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> getPixelPatchFeatures(FImage image) {
        LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>> features = new MemoryLocalFeatureList<>();
        //create 8x8 pixel patches sampled every 4 pixels in the x and y directions
        RectangleSampler sampler = new RectangleSampler(image.normalise(),4,4,8,8);
        List<Rectangle> rectangles = sampler.allRectangles();
        for(Rectangle r : rectangles) {
            //create an image from the given patch, then normalise it
            FImage rImg = image.extractROI(r).normalise();
            //flatten patch into vector
            float[] rVector = rImg.getFloatPixelVector();
            //create local feature from vector and location
            SpatialLocation rLoc = new SpatialLocation(r.x, r.y);
            features.add(new LocalFeatureImpl<>(rLoc, new FloatFV(rVector)));
        }
        return features;
    }

    /**
     * Create a GroupedDataset from a VFSGroupDataset by creating a Record from each image and its class
     * @param originalDataset - the dataset that should be converted
     * @return a GroupedDataset
     */
    public static GroupedDataset<String, ListBackedDataset<Record>, Record> convertToGroupedDataset(VFSGroupDataset<FImage> originalDataset) {
        GroupedDataset<String, ListBackedDataset<Record>, Record> recordDataset = new MapBackedDataset<>();
        //iterate through each class in original dataset
        for (final Map.Entry<String, VFSListDataset<FImage>> entry : originalDataset.entrySet()) {
            //create a ListBackedDataset of each image class
            VFSListDataset<FImage> imageList = entry.getValue();
            ListBackedDataset<Record> recordList = new ListBackedDataset<>();
            for (int i = 0; i < imageList.size(); i++) {
                FImage image = imageList.get(i);
                recordList.add(new Record(image, i+"", entry.getKey()));
            }
            //add to GroupedDataset
            recordDataset.put(entry.getKey(), recordList);
        }
        return recordDataset;
    }
}
