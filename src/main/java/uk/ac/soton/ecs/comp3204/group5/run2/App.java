package uk.ac.soton.ecs.comp3204.group5.run2;

import de.bwaldvogel.liblinear.SolverType;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class App {
    public static void main( String[] args ) throws FileSystemException {
        VFSGroupDataset<FImage> originalDataset = new VFSGroupDataset<>( "C:\\Users\\Akhilesh\\Downloads\\training", ImageUtilities.FIMAGE_READER);
        GroupedDataset<String, ListBackedDataset<Record>, Record> recordDataset = new MapBackedDataset<>();

        for (final Map.Entry<String, VFSListDataset<FImage>> entry : originalDataset.entrySet()) {
            VFSListDataset<FImage> imageList = entry.getValue();
            ListBackedDataset<Record> recordList = new ListBackedDataset<>();
            for (int i = 0; i < imageList.size(); i++) {
                FImage image = imageList.get(i);
                recordList.add(new Record(image, i+"", entry.getKey()));
            }

            recordDataset.put(entry.getKey(), recordList);
        }

        GroupedDataset<String, ListDataset<Record>, Record> sampledData = GroupSampler.sample(recordDataset, 5, false);

        GroupedRandomSplitter<String, Record> splitData = new GroupedRandomSplitter<String, Record>(sampledData, 15, 0, 15);

        //float[][] patchVectors = new float[][];
        //iterate through samples, create fixed size densely-sampled pixel patches from their normalised form
        for(Record record : GroupedUniformRandomisedSampler.sample(splitData.getTrainingDataset(), 30)) {
            RectangleSampler sampler = new RectangleSampler(record.getImage().normalise(),4,4,8,8);
            List<Rectangle> patchesOfRecord = sampler.allRectangles();
            //take the pixels from the patches, flatten them into a vector, then add the vector to a list
            for(Rectangle patch : patchesOfRecord) {
                FImage img = record.getImage().normalise().extractROI(patch);
                float[] imgVector = img.getFloatPixelVector();
                /*add this vector to a data structure
                * eventually a float[][] needs to be made to create the assigner
                 */
            }
        }
        //from the patch vectors created above, create an assigner
        //HardAssigner<float[], float[], IntFloatPair> assigner = createVocabulary(patchVectors);
        //use assigner to create a FeatureExtractor
        //FeatureExtractor<DoubleFV,Record> extractor = new Extractor(assigner);
        //use FeatureExtractor to create classifier
        //LiblinearAnnotator<Record, String> ann = new LiblinearAnnotator<Record, String>(extractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        //train classifier on dataset
        //test classifier
    }

    /**
     * Learn a vocabulary by clustering a sample of pixel patches
     * @param patches - an array of float vectors of patches of sample images
     * @return - a default Hard Assigner created from applying Kmeans to the input
     */
    static HardAssigner<float[], float[], IntFloatPair> createVocabulary(float[][] patches) {
        FloatKMeans kMeans = FloatKMeans.createExact(500);
        FloatCentroidsResult result = kMeans.cluster(patches);
        return result.defaultHardAssigner();
    }

    /**
     * WIP class for a Feature Extractor which uses a HardAssigner to create a BagOfVisualWords feature
     */
    static class Extractor implements FeatureExtractor<DoubleFV, Record> {

        HardAssigner<float[], float[], IntFloatPair> assigner;
        Extractor(HardAssigner<float[], float[], IntFloatPair> assigner) {
            this.assigner = assigner;
        }

        @Override
        public DoubleFV extractFeature(Record record) {
            FImage image = record.getImage();
            BagOfVisualWords bagOfVisualWords = new BagOfVisualWords(assigner);
            //do something
            return null;
        }
    }

}
