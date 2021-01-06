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
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import uk.ac.soton.ecs.comp3204.group5.Helper;
import uk.ac.soton.ecs.comp3204.group5.Record;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * COMP3204 - Computer Vision
 * Code for Run 1 - KNNAnnotator using Tiny Images
 */
public class App {
    private static final String filepathTraining = "C:\\Users\\Nawab\\Downloads\\training";
    private static final String filepathTesting = "C:\\Users\\Nawab\\Downloads\\testing";
    private static final String filepathOutput = "C:\\Users\\Nawab\\Downloads\\run1.txt";

    /**
     * This function generates 10 different splits of data for k values up to 30 and evaluates the performance
     *  of the classifier
     * */
    public static void testEnvironment() throws Exception {
        TinyImageExtractor tinyImageExtractor = new TinyImageExtractor(16);

        VFSGroupDataset<FImage> input = new VFSGroupDataset<FImage>(filepathTraining, ImageUtilities.FIMAGE_READER);

        for(int k = 1; k < 30; k++) {
            System.out.println("EVALUATE CLASSIFIER K = " + k);
            double result = 0;
            for(int i = 1; i < 10; i++) {
                // Use GroupedRandomSplitter to generate train and test dataset
                GroupedRandomSplitter<String, Record> splitter = new GroupedRandomSplitter(input,75,0,25);
                // Generator KNN annotator which uses the euclidean distance between images
                KNNAnnotator<Record, String, TinyImageExtractor> knnAnnotator = new KNNAnnotator(tinyImageExtractor, DoubleFVComparison.EUCLIDEAN, k);

                // Train the annotator on the training data
                knnAnnotator.trainMultiClass(splitter.getTrainingDataset());

                // Evaluate the outcome of the test data set
                ClassificationEvaluator<CMResult<String>, String, Record> eval =
                        new ClassificationEvaluator<>(
                                knnAnnotator, splitter.getTestDataset(), new CMAnalyser<Record, String>(CMAnalyser.Strategy.SINGLE));

                Map<Record, ClassificationResult<String>> guesses = eval.evaluate();
                result += eval.analyse(guesses).getMatrix().getAccuracy();

            }
            // Average the accuracy
            System.out.println(result / 10);
        }
    }

    public static void main(String[] args) throws Exception {
        // Feature Extractor for TinyImages
        TinyImageExtractor tinyImageExtractor = new TinyImageExtractor(16);

        // Import training and testing data
        VFSGroupDataset<FImage> training = new VFSGroupDataset<FImage>(filepathTraining, ImageUtilities.FIMAGE_READER);

        // Create KNN Annotator using Tiny Image feature extractor using EUCLIDEAN FV comparator and k = 15
        // k = 15 was learned as the best value of K using a 75/25 split from the training data
        KNNAnnotator<FImage, String, TinyImageExtractor> knnAnnotator = new KNNAnnotator(tinyImageExtractor, DoubleFVComparison.EUCLIDEAN, 15);

        // Train annotator on training data
        knnAnnotator.trainMultiClass(training);
        testEnvironment();
        // Annotate the unlabelled testing data and output the best value for each image
        /** This is equivalent to Helper::makePredictions */

        BufferedWriter writer = new BufferedWriter(new FileWriter(filepathOutput));
        VFSListDataset<FImage> testing = new VFSListDataset<FImage>(filepathTesting, ImageUtilities.FIMAGE_READER);

        for (int i = 0; i < testing.size(); i++) {
            List<ScoredAnnotation<String>> result = knnAnnotator.annotate(testing.get(i));
            Collections.sort(result, Collections.reverseOrder());
            String outputRes = testing.getID(i) + " ";
            outputRes += result.get(0).annotation;
            writer.write(outputRes);
            writer.newLine();
        }
        writer.close();

    }
}
