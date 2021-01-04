package uk.ac.soton.ecs.comp3204.group5.run1;

import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.basic.KNNAnnotator;
import uk.ac.soton.ecs.comp3204.group5.Helper;
import uk.ac.soton.ecs.comp3204.group5.Record;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class App2 {

    public void testEnvironment() throws Exception {
        TinyImageExtractor tinyImageExtractor = new TinyImageExtractor(16);

        VFSGroupDataset<FImage> input = new VFSGroupDataset<FImage>("C:\\Users\\Nawab\\Downloads\\training", ImageUtilities.FIMAGE_READER);

        // Use GroupedRandomSplitter to generate train and test dataset
        GroupedRandomSplitter<String, Record> splitter = new GroupedRandomSplitter(input,75,0,25);


        for(int k = 1; k < 20; k++) {
            // Generator KNN annotator which uses the euclidean distance between images
            KNNAnnotator<Record, String, TinyImageExtractor> knnAnnotator = new KNNAnnotator(tinyImageExtractor, DoubleFVComparison.EUCLIDEAN, k);

            // Train the annotator on the training data
            knnAnnotator.trainMultiClass(splitter.getTrainingDataset());

            // Evaluate the outcome of the test data set
            ClassificationEvaluator<CMResult<String>, String, Record> eval =
                    new ClassificationEvaluator<>(
                            knnAnnotator, splitter.getTestDataset(), new CMAnalyser<Record, String>(CMAnalyser.Strategy.SINGLE));

            System.out.println("EVALUATE CLASSIFIER K = " + k);
            Map<Record, ClassificationResult<String>> guesses = eval.evaluate();
            CMResult<String> result = eval.analyse(guesses);

            System.out.println(result);
        }
    }

    public static void classifyTestData() throws Exception {
        // Feature Extractor for TinyImages
        TinyImageExtractor tinyImageExtractor = new TinyImageExtractor(16);

        // Import training and testing data
        VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>("/Users/shintaroonuma/Documents/cwImages/training", ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testing = new VFSListDataset<FImage>("/Users/shintaroonuma/Documents/cwImages/testing", ImageUtilities.FIMAGE_READER);

        // Create KNN Annotator using Tiny Image feature extractor using EUCLIDEAN FV comparator and k = 15
        // k = 15 was learned as the best value of K using a 75/25 split from the training data
        KNNAnnotator<FImage, String, TinyImageExtractor> knnAnnotator = new KNNAnnotator(tinyImageExtractor, DoubleFVComparison.EUCLIDEAN, 15);

        // Train annotator on training data
        knnAnnotator.trainMultiClass(training);

        // Annotate the unlabelled testing data and output the best value for each image
        for (int i = 0; i < testing.size(); i++) {
            List<ScoredAnnotation<String>> result = knnAnnotator.annotate(testing.get(i));
            Collections.sort(result, Collections.reverseOrder());
            String outputRes = testing.getID(i) + " ";
            outputRes += result.get(0).annotation;
            System.out.println(outputRes);
        }
    }
}
