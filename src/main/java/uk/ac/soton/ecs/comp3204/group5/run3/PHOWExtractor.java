package uk.ac.soton.ecs.comp3204.group5.run3;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.group5.Record;

/**
 * Pyramid Histogram of Words implementation for run 3
 */
public class PHOWExtractor implements FeatureExtractor<DoubleFV,Record> {
    PyramidDenseSIFT<FImage> pdsift;
    HardAssigner<float[], float[], IntFloatPair> assigner;

    public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<float[], float[], IntFloatPair> assigner) {
        this.pdsift = pdsift;
        this.assigner = assigner;
    }

    @Override
    public DoubleFV extractFeature(Record record) {
        FImage image = record.getImage();
        pdsift.analyseImage(image);
        BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<>(assigner);
        PyramidSpatialAggregator<float[], SparseIntFV> spatial =
                new PyramidSpatialAggregator<>(bagOfVisualWords, 2, 4);
        return spatial.aggregate(pdsift.getFloatKeypoints(0.015f), image.getBounds()).normaliseFV();
    }
}
