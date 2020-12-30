package uk.ac.soton.ecs.comp3204.group5.run2;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.group5.Helper;
import uk.ac.soton.ecs.comp3204.group5.Record;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class App {
    private static final String filepath1 = "C:\\Users\\peter\\Downloads\\training";
    public static void main( String[] args ) throws FileSystemException {
        //load dataset from file
        VFSGroupDataset<FImage> originalDataset = new VFSGroupDataset<>( filepath1, ImageUtilities.FIMAGE_READER);
        System.out.println("LOADED DATASET");
        //convert to GroupedDataset to provide more functionality
        GroupedDataset<String, ListBackedDataset<Record>, Record> recordDataset = Helper.convertToGroupedDataset(originalDataset);
        System.out.println("POPULATED RECORD DATASET");
        //GroupedDataset<String, ListDataset<Record>, Record> sampledData = GroupSampler.sample(recordDataset, 5, false);

        GroupedRandomSplitter<String, Record> splitData = new GroupedRandomSplitter<>(recordDataset, 80, 0, 20);
        System.out.println("SPLIT DATASET INTO TRAINING AND TESTING");

        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(GroupedUniformRandomisedSampler.sample(splitData.getTrainingDataset(), 30));
        System.out.println("CLUSTERING DONE");
        // use assigner to create a FeatureExtractor
        FeatureExtractor<SparseIntFV,Record> extractor = new BOVWExtractor(assigner);
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

    /**
     * Creates a vocabulary of visual words from a sample of images using K-means clustering
     * @param sample - a subset of the training dataset
     * @return a vocabulary of visual words
     */
    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(GroupedDataset<String, ListDataset<Record>, Record> sample) {
        List<LocalFeatureList<LocalFeatureImpl<SpatialLocation, FloatFV>>> allKeys = new ArrayList<>();
        for(Record record : sample) {
            allKeys.add(Helper.getPixelPatchFeatures(record.getImage()));
        }

        if (allKeys.size() > 10000) {
            allKeys = allKeys.subList(0, 10000);
        }
        FloatKMeans kMeans = FloatKMeans.createExact(600);
        DataSource<float[]> dataSource = new LocalFeatureListDataSource<>(allKeys);
        FloatCentroidsResult result = kMeans.cluster(dataSource);
        return  result.defaultHardAssigner();
    }

    /**
     * Implementation of a Feature Extractor which uses a HardAssigner to create a BagOfVisualWords feature
     * which extracts features from densely sampled pixel patches
     */
    static class BOVWExtractor implements FeatureExtractor<SparseIntFV, Record> {
        BagOfVisualWords<float[]> bagOfVisualWords;
        HardAssigner<float[], float[], IntFloatPair> assigner;
        public BOVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.assigner = assigner;
            this.bagOfVisualWords = new BagOfVisualWords<>(assigner);
        }

        @Override
        public SparseIntFV extractFeature(Record record) {
            FImage image = record.getImage();
            return bagOfVisualWords.aggregate(Helper.getPixelPatchFeatures(image));
        }
    }
}
