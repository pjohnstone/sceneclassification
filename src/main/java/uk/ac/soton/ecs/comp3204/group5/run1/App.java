package uk.ac.soton.ecs.comp3204.group5.run1;


import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * COMP3204 Group 5 Run 1
 *
 *
 */
public class App {
	
	static DoubleNearestNeighboursExact knn;
	static ArrayList<String> classes;

	/**
	 * Creates tiny image feature from FImage parameter
	 *
	 */
	public static double[] getTinyImageFeature(FImage input) {
		//get smallest side and crop image so both width and height are equal to smallest side
		int minSide = Math.min(input.height , input.width);
		FImage cropped = input.extractCenter(minSide, minSide);

		//resizes image to 16 by 16 pixels and normalises this image
		FImage processed = ResizeProcessor.resample(cropped, 16, 16).normalise();

		//concatenates image rows into double vector
		double[] output = processed.getDoublePixelVector();
		return output;
	}

	/**
	 * trains K Nearest Neighbours and stores this in global "knn" variable
	 * - uses ListDataset instead of VFSListDataset
	 * - stores the image classes in the global arraylist so we can retrieve the neighbour classes when we test our model
	 */
	public static void trainKNN(GroupedDataset<String, ListDataset<FImage>, FImage> training) {
		classes = new ArrayList<String>();
		ArrayList<double[]> vectors = new ArrayList<double[]>();
		//iterate through all training images and generate tiny image feature vectors
		for(final Entry<String, ListDataset<FImage>> entry : training.entrySet()){
			for(FImage img : entry.getValue()) {
				classes.add(entry.getKey());
				vectors.add(getTinyImageFeature(img));
			}
		}
		//use constructor with feature vector array to train knn model
		knn = new DoubleNearestNeighboursExact(vectors.toArray(new double[][]{}));
	}
	/**
	 * trains K Nearest Neighbours and stores this in global "knn" variable
	 * - uses VFSListDataset instead of ListDataset
	 * - stores the image classes in the global arraylist so we can retrieve the neighbour classes when we test our model
	 */
	private static void VFStrainKNN(GroupedDataset<String, VFSListDataset<FImage>, FImage> training) {
		classes = new ArrayList<String>();
		ArrayList<double[]> vectors = new ArrayList<double[]>();
		//iterate through all training images and generate tiny image feature vectors
		for(final Entry<String, VFSListDataset<FImage>> entry : training.entrySet()){
			for(FImage img : entry.getValue()) {
				classes.add(entry.getKey());
				vectors.add(getTinyImageFeature(img));
			}
		}
		//use constructor with feature vector array to train knn model
		knn = new DoubleNearestNeighboursExact(vectors.toArray(new double[][]{}));
	}

	/**
	 * tests accuracy of K nearest neighbour classifier
	 *
	 */
	public static double testKNN(GroupedDataset<String, ListDataset<FImage>, FImage> testing , int k, boolean printResults) {
		int total = 0;
		int correct = 0;
		//iterate through all images in test dataset
		for(final Entry<String, ListDataset<FImage>> entry : testing.entrySet()){
			for(FImage img : entry.getValue()) {
				//searches for k nearest neighbours
				List<IntDoublePair> result = knn.searchKNN(getTinyImageFeature(img), k);

				//store each neighbour class into a map and retrieve most common class
				Map<String,Integer> tally = new HashMap<String,Integer>();
				for(IntDoublePair pair : result) {
					String name = classes.get(pair.getFirst());
					if(tally.containsKey(name)) {
						tally.put(name, tally.get(name)+1);
					}
					else {
						tally.put(name,1);
					}
				}
				String finalRes = "";
				int best = 0;
				for(Map.Entry<String, Integer> e : tally.entrySet()) {
					if(best < e.getValue()) {
						finalRes = e.getKey();
						best = e.getValue();
					}
				}

				total++;
				//if the most common result is equal to the expected class then increment correct total
				if(finalRes == entry.getKey()) {
					correct++;
				}
			}
		}
		//calculate accuracy percentage
		double accuracy = ((double)correct / (double)total ) * 100;
		if(printResults) {
			System.out.println("correct: " + correct);
			System.out.println("total: "  + total);
			System.out.println("accuracy: " + accuracy);
		}
		return accuracy;
	}

	/**
	 * classify unlabelled dataset using knn classifier
	 *
	 */
	public static ArrayList<String> classKNN(VFSListDataset<FImage> testing , int k) {
		ArrayList<String> classifications = new ArrayList<String>();
		//iterate through each image using index for easy image ID retrieval
		for(int i = 0 ; i < testing.size() ; i++) {
			FImage img = testing.get(i);
			//get k nearest neighbours of current image feature vector
			List<IntDoublePair> result = knn.searchKNN(getTinyImageFeature(img), k);

			//use map to retrieve most common class of neighbours
			Map<String,Integer> tally = new HashMap<String,Integer>();
			for(IntDoublePair pair : result) {
				String name = classes.get(pair.getFirst());
				if(tally.containsKey(name)) {
					tally.put(name, tally.get(name)+1);
				}
				else {
					tally.put(name,1);
				}
			}
			String finalRes = "";
			int best = 0;
			for(Map.Entry<String, Integer> entry : tally.entrySet()) {
				if(best < entry.getValue()) {
					finalRes = entry.getKey();
					best = entry.getValue();
				}
			}
			//generate "<image_name> <predicted_class>" string
			String outputRes = testing.getID(i) + " ";
			outputRes += finalRes;
			classifications.add(outputRes);
		}
		return classifications;
	}

	/**
	 * run function for predicting classes for test data
	 * prints classification results in required format
	 */
	public static void classificationRun(){
		try {
			//get training and testing dataset using filepaths
			GroupedDataset<String, VFSListDataset<FImage>, FImage> train =new VFSGroupDataset<FImage>("/Users/shintaroonuma/Documents/cwImages/training", ImageUtilities.FIMAGE_READER);
			VFSListDataset<FImage> test = new VFSListDataset<FImage>("/Users/shintaroonuma/Documents/cwImages/testing", ImageUtilities.FIMAGE_READER);
    		//GroupedDataset<String, ListDataset<FImage>, FImage> trainNewFormat = splitter.getTrainingDataset();

			//train KNN using training dataset
    		VFStrainKNN(train);

    		//get and print classification results of training dataset
    		ArrayList<String> ans = classKNN(test,1);
    		for(String s : ans) {
    			System.out.println(s);
    		}

    	} catch (FileSystemException e) {
			e.printStackTrace();
		}
	}


	/**
	 * run function for calculating accuracy values for various k values for our nearest neighbour classfication
	 *
	 */
	public static void accuracyRun(){
		try {
			
			VFSGroupDataset<FImage> input = new VFSGroupDataset<FImage>("/Users/shintaroonuma/Documents/cwImages/training", ImageUtilities.FIMAGE_READER);
			
			int iter = 20;
			double a1 = 0;
			double a2 = 0;
			double a3 = 0;
			double a4 = 0;
			double a5 = 0;
			double a6 = 0;
			double a7 = 0;
			double a8 = 0;
			double a9 = 0;
			double a10 = 0;
			//generate new randomly split dataset and calculate accuracies for each k value
			for(int i = 0 ; i < iter ; i++) {
				//use GroupedRandomSplitter to generate train and test dataset
				GroupedRandomSplitter<String,FImage> splitter = new GroupedRandomSplitter(input,80,0,20);
	    		GroupedDataset<String, ListDataset<FImage>, FImage> train = splitter.getTrainingDataset();
	    		GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();
	    		//train new knn classifier using train dataset
	    		trainKNN(train);
	    		//add accuracy percentage for each k value to respective variables
	    		a1 += testKNN(test,1,false);
	    		a2 += testKNN(test,2,false);
	    		a3 += testKNN(test,3,false);
	    		a4 += testKNN(test,4,false);
	    		a5 += testKNN(test,5,false);
	    		a6 += testKNN(test,6,false);
	    		a7 += testKNN(test,7,false);
	    		a8 += testKNN(test,8,false);
	    		a9 += testKNN(test,9,false);
	    		a10 += testKNN(test,10,false);
			}
			a1 = a1 / iter;
			a2 = a2/iter;
			a3 = a3 / iter;
			a4 = a4/iter;
			a5 = a5/iter;
			a6 = a6/iter;
			a7 = a7 / iter;
			a8 = a8/iter;
			a9 = a9/iter;
			a10 = a10/iter;
			
			System.out.format("k = 1: %.2f  ", a1);
			System.out.format("k = 2: %.2f  ", a2);
			System.out.format("k = 3: %.2f  ", a3);
			System.out.format("k = 4: %.2f  ", a4);
			System.out.format("k = 5: %.2f  ", a5);
			System.out.format("k = 6: %.2f  ", a6);
			System.out.format("k = 7: %.2f  ", a7);
			System.out.format("k = 8: %.2f  ", a8);
			System.out.format("k = 9: %.2f  ", a9);
			System.out.format("k = 10: %.2f  ", a10);
    	} catch (FileSystemException e) {
			e.printStackTrace();
		}
	}
	
	
    public static void main( String[] args ) {
    	//classificationRun();
    	accuracyRun();
    }
}

