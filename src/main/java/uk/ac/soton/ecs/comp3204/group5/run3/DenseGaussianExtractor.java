package uk.ac.soton.ecs.comp3204.group5.run3;

import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.group5.Record;

public class DenseGaussianExtractor implements FeatureExtractor<SparseIntFV,Record> {
    PyramidDenseSIFT<FImage> pdsift;
    HardAssigner<float[], float[], IntFloatPair> assigner;
    BagOfVisualWords<float[]> bagOfVisualWords;

    public DenseGaussianExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<float[], float[], IntFloatPair> assigner) {
        this.pdsift = pdsift;
        this.assigner = assigner;
        this.bagOfVisualWords = new BagOfVisualWords<>(assigner);
    }

    @Override
    public SparseIntFV extractFeature(Record record) {
        FImage image = record.getImage();
        BlockSpatialAggregator<float[], SparseIntFV> spatial =
                new BlockSpatialAggregator<>(this.bagOfVisualWords, 2, 2);

        return spatial.aggregate(pdsift.getFloatKeypoints(0.015f), image.getBounds());
    }
}
