package uk.ac.soton.ecs.comp3204.group5.run3;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.FloatDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.group5.Helper;
import uk.ac.soton.ecs.comp3204.group5.Record;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * COMP3204 - Computer Vision
 * Code for Run 3 - Scene recognition using Pyramid Dense SIFT
 */
public class App {
    private static final String filepath = "";
    public static void main( String[] args ) throws FileSystemException {
        // Load dataset into GroupedDataset of Records
        VFSGroupDataset<FImage> originalDataset = new VFSGroupDataset<>(filepath, ImageUtilities.FIMAGE_READER);
        GroupedDataset<String, ListBackedDataset<Record>, Record> recordDataset = Helper.convertToGroupedDataset(originalDataset);
        GroupedRandomSplitter<String, Record> splitData = new GroupedRandomSplitter<>(recordDataset, 80, 0, 20);

        // Create Dense SIFT in Gaussian Pyramid
        DenseSIFT dsift = new DenseSIFT(3, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 4, 6, 8, 10);

        // Train the quantiser on a sample of the data
        HardAssigner<float[], float[], IntFloatPair> assigner =
                trainQuantiser(GroupedUniformRandomisedSampler.sample(splitData.getTrainingDataset(), 45), pdsift);

        // Create set of linear classifiers and train the LiblinearAnnotator wrapped in a Homogeneous Kernel Map
        HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor<DoubleFV, Record> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));
        LiblinearAnnotator<Record, String> ann = new LiblinearAnnotator<>(
                extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        ann.train(splitData.getTrainingDataset());

        // Classify the test data and print the accuracy
        ClassificationEvaluator<CMResult<String>, String, Record> eval =
                new ClassificationEvaluator<>(
                        ann, splitData.getTestDataset(), new CMAnalyser<Record, String>(CMAnalyser.Strategy.SINGLE));

        Map<Record, ClassificationResult<String>> guesses = eval.evaluate();
        CMResult<String> result = eval.analyse(guesses);

        System.out.println(result);

        // To run the predictions, comment out the below code and train
        // the annotator on the whole dataset rather than a test-train split
        // Helper.makePredictions(ann, testFilepath, outputFilepath );
    }

    /**
     * Creates a vocabulary of visual words from a sample of images using K-means clustering of Dense SIFT patches
     * @param sample - a subset of the training dataset
     * @return a vocabulary of visual words
     */
    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<Record> sample, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<FloatDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (Record rec : sample) {
            FImage img = rec.getImage();

            pdsift.analyseImage(img);
            allkeys.add(pdsift.getFloatKeypoints(0.005f));
        }

        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(300);
        DataSource<float[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        FloatCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }
}
