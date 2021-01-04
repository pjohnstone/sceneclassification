package uk.ac.soton.ecs.comp3204.group5.run3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.FloatDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.model.EigenImages;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;
import uk.ac.soton.ecs.comp3204.group5.Helper;
import uk.ac.soton.ecs.comp3204.group5.Record;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class AppPCA {

    private static final String filepath2 = "C:\\Users\\Akhilesh\\Downloads\\training";

    public static void main( String[] args ) throws FileSystemException {
        VFSGroupDataset<FImage> originalDataset = new VFSGroupDataset<>(filepath2, ImageUtilities.FIMAGE_READER);
        GroupedDataset<String, ListBackedDataset<Record>, Record> recordDataset = Helper.convertToGroupedDataset(originalDataset);
        int nTraining = 80;
        int nTesting = 20;
        GroupedRandomSplitter<String, Record> splitData =
                new GroupedRandomSplitter<>(recordDataset, nTraining, 0, nTesting);
        GroupedDataset<String, ListDataset<Record>, Record> training = splitData.getTrainingDataset();
        GroupedDataset<String, ListDataset<Record>, Record> testing = splitData.getTestDataset();
        DenseSIFT dsift = new DenseSIFT(3, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 4, 6, 8, 10);

        System.out.println("DATA SPLIT");

        HardAssigner<float[], float[], IntFloatPair> assigner =
                trainQuantiser(GroupedUniformRandomisedSampler.sample(splitData.getTrainingDataset(), 45), pdsift);

        System.out.println("HARD ASSIGNER MADE");

        HomogeneousKernelMap hkm = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor<DoubleFV, Record> extractor = hkm.createWrappedExtractor(new PHOWExtractor(pdsift, assigner));

        System.out.println("FEATURE EXTRACTOR MADE");

        List<Record> basisImages = DatasetAdaptors.asList(training);
        int nEigenvectors = 100;
        EigenScenes eigen = new EigenScenes(nEigenvectors, extractor);
        eigen.train((ArrayList<Record>) basisImages);

        Map<String, DoubleFV[]> features = new HashMap<>();
        for (final String person : training.getGroups()) {
            final DoubleFV[] fvs = new DoubleFV[nTraining];
            for (int i = 0; i < nTraining; i++) {
                final Record face = training.get(person).get(i);
                fvs[i] = eigen.extractFeature(face);
            }
            features.put(person, fvs);
        }

        System.out.println("FEATURES PUT");

        double correct = 0, incorrect = 0;
        for (String truePerson : testing.getGroups()) {
            System.out.println("CURRENT SCENE " + truePerson);
            for (Record face : testing.get(truePerson)) {
                DoubleFV testFeature = eigen.extractFeature(face);
                String bestPerson = null;
                double minDistance = Double.MAX_VALUE;
                for (final String person : features.keySet()) {
                    for (final DoubleFV fv : features.get(person)) {
                        double distance = fv.compare(testFeature, DoubleFVComparison.EUCLIDEAN);
                        if (distance < minDistance) {
                            minDistance = distance;
                            bestPerson = person;
                        }
                    }
                }
                System.out.println("Actual: " + truePerson + "\tguess: " + bestPerson);
                if (truePerson.equals(bestPerson))
                    correct++;
                else
                    incorrect++;
            }
        }
        System.out.println("Accuracy: " + (correct / (correct + incorrect)));
    }

    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(
            Dataset<Record> sample, PyramidDenseSIFT<FImage> pdsift)
    {
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
