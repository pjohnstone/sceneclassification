package uk.ac.soton.ecs.comp3204.group5.run2;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.image.feature.local.keypoints.FloatKeypoint;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.Helper;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class App {
    private static final String filepath1 = "C:\\Users\\peter\\Downloads\\training";
    public static void main( String[] args ) throws FileSystemException {

        VFSGroupDataset<FImage> originalDataset = new VFSGroupDataset<>( filepath1, ImageUtilities.FIMAGE_READER);
        System.out.println("LOADED DATASET");
        GroupedDataset<String, ListBackedDataset<Record>, Record> recordDataset = Helper.convertToGroupedDataset(originalDataset);
        System.out.println("POPULATED RECORD DATASET");

        //GroupedDataset<String, ListDataset<Record>, Record> sampledData = GroupSampler.sample(recordDataset, 5, false);

        GroupedRandomSplitter<String, Record> splitData = new GroupedRandomSplitter<>(recordDataset, 80, 0, 20);

        System.out.println("SAMPLED DATASET");

        List<float[]> vectorList = new ArrayList<>();
        List<FloatKeypoint> floatKeypoints = new ArrayList<>();
        // iterate through samples, create fixed size densely-sampled pixel patches from their normalised form
        for(Record record : GroupedUniformRandomisedSampler.sample(splitData.getTrainingDataset(), 60)) {
            // mean centre
            RectangleSampler sampler = new RectangleSampler(record.getImage().normalise(),4,4,8,8);
            List<Rectangle> patchesOfRecord = sampler.allRectangles();
            // take the pixels from the patches, flatten them into a vector, then add the vector to a list
            for(Rectangle patch : patchesOfRecord) {
                FImage img = record.getImage().normalise().extractROI(patch);
                float[] imgVector = img.getFloatPixelVector();
                vectorList.add(imgVector);
                floatKeypoints.add(new FloatKeypoint(patch.x, patch.y, 0, 1, imgVector));
            }
        }
        float[][] patchVectors = convertToArr(vectorList); System.out.println("CREATED PATCHES");
        // from the patch vectors created above, create an assigner
        HardAssigner<float[], float[], IntFloatPair> assigner = createVocabulary(patchVectors); System.out.println("CLUSTERED PATCHES");
        // use assigner to create a FeatureExtractor
        FeatureExtractor<SparseIntFV,Record> extractor = new Extractor(assigner); System.out.println("EXTRACT FEATURES");
        // use FeatureExtractor to create classifier
        LiblinearAnnotator<Record, String> ann =
                new LiblinearAnnotator<>(extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        // train classifier on dataset
        System.out.println("TRAIN CLASSIFIER");
        ann.train(splitData.getTrainingDataset());
        ClassificationEvaluator<CMResult<String>, String, Record> eval =
                new ClassificationEvaluator<>(
                        ann, splitData.getTestDataset(), new CMAnalyser<Record, String>(CMAnalyser.Strategy.SINGLE));
        System.out.println("EVALUATE CLASSIFIER");
        Map<Record, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result);
    }

    private static float[][] convertToArr(List<float[]> vectors) {
        float[][] floatArr = new float[vectors.size()][];
        for(int i = 0; i < vectors.size(); i++) {
            floatArr[i] = vectors.get(i);
        }
        return floatArr;
    }

    /**
     * Learn a vocabulary by clustering a sample of pixel patches
     * @param patches - an array of float vectors of patches of sample images
     * @return - a default Hard Assigner created from applying Kmeans to the input
     */
    static HardAssigner<float[], float[], IntFloatPair> createVocabulary(float[][] patches) {
        FloatKMeans kMeans = FloatKMeans.createExact(600);
        FloatCentroidsResult result = kMeans.cluster(patches);
        return result.defaultHardAssigner();
    }

    /**
     * WIP class for a Feature Extractor which uses a HardAssigner to create a BagOfVisualWords feature
     */
    static class Extractor implements FeatureExtractor<SparseIntFV, Record> {
        BagOfVisualWords<float[]> bagOfVisualWords;
        HardAssigner<float[], float[], IntFloatPair> assigner;
        public Extractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.assigner = assigner;
            this.bagOfVisualWords = new BagOfVisualWords<>(assigner);
        }

        @Override
        public SparseIntFV extractFeature(Record record) {
            FImage image = record.getImage();
            return bagOfVisualWords.aggregate(Helper.createLocalFeatureList(image));
        }
    }
}
