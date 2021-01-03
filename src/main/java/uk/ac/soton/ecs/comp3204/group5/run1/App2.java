package uk.ac.soton.ecs.comp3204.group5.run1;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import uk.ac.soton.ecs.comp3204.group5.Record;

import java.util.Map;

public class App2 {

    public static void main(String[] args) throws Exception{
        TinyImageExtractor tinyImageExtractor = new TinyImageExtractor(16);

        VFSGroupDataset<FImage> input = new VFSGroupDataset<FImage>("C:\\Users\\Nawab\\Downloads\\training", ImageUtilities.FIMAGE_READER);

        // Use GroupedRandomSplitter to generate train and test dataset
        GroupedRandomSplitter<String, Record> splitter = new GroupedRandomSplitter(input,80,0,20);


        for(int k = 1; k < 20; k++) {
            // Generator KNN annotator which uses the euclidean distance between images
            KNNAnnotator<Record, String, TinyImageExtractor> knnAnnotator = new KNNAnnotator(tinyImageExtractor, DoubleFVComparison.EUCLIDEAN, k);

            // Train the annotator on the training data
            knnAnnotator.trainMultiClass(splitter.getTrainingDataset());

            // Evaluate the outcome of the test data set
            ClassificationEvaluator<CMResult<String>, String, Record> eval =
                    new ClassificationEvaluator<>(
                            knnAnnotator, splitter.getTestDataset(), new CMAnalyser<Record, String>(CMAnalyser.Strategy.SINGLE));

            System.out.println("EVALUATE CLASSIFIER " + k);
            Map<Record, ClassificationResult<String>> guesses = eval.evaluate();
            CMResult<String> result = eval.analyse(guesses);

            System.out.println(result);
        }
    }
}
