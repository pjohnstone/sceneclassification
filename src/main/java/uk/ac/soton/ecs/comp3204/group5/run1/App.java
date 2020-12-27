package uk.ac.soton.ecs.comp3204.group5.run1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

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
		double[] output = new double[0];
		//get smallest side and crop image so both width and height are equal to smallest side
		int minSide = Math.min(input.height , input.width);
		FImage cropped = input.extractCenter(minSide, minSide);

		FImage processed = ResizeProcessor.resample(cropped, 16, 16).normalise();
		output = processed.getDoublePixelVector();
		return output;
	}
	
	public static void trainKNN(GroupedDataset<String, ListDataset<FImage>, FImage> training) {
		classes = new ArrayList<String>();
		ArrayList<double[]> vectors = new ArrayList<double[]>();
		
		for(final Entry<String, ListDataset<FImage>> entry : training.entrySet()){
			for(FImage img : entry.getValue()) {
				classes.add(entry.getKey());
				vectors.add(getTinyImageFeature(img));
			}
		}
		knn = new DoubleNearestNeighboursExact(vectors.toArray(new double[][]{}));
	}
	
	private static void VFStrainKNN(GroupedDataset<String, VFSListDataset<FImage>, FImage> training) {
		classes = new ArrayList<String>();
		ArrayList<double[]> vectors = new ArrayList<double[]>();
		
		for(final Entry<String, VFSListDataset<FImage>> entry : training.entrySet()){
			for(FImage img : entry.getValue()) {
				classes.add(entry.getKey());
				vectors.add(getTinyImageFeature(img));
			}
		}
		knn = new DoubleNearestNeighboursExact(vectors.toArray(new double[][]{}));
	}
	
	public static double testKNN(GroupedDataset<String, ListDataset<FImage>, FImage> testing , int k, boolean printResults) {
		int total = 0;
		int correct = 0;
		for(final Entry<String, ListDataset<FImage>> entry : testing.entrySet()){
			for(FImage img : entry.getValue()) {
				List<IntDoublePair> result = knn.searchKNN(getTinyImageFeature(img), k);
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
				if(finalRes == entry.getKey()) {
					correct++;
				}
			}
		}
		double accuracy = ((double)correct / (double)total ) * 100;
		if(printResults) {
			System.out.println("correct: " + correct);
			System.out.println("total: "  + total);
			System.out.println("accuracy: " + accuracy);
		}
		return accuracy;
	}
	
	public static ArrayList<String> classKNN(VFSListDataset<FImage> testing , int k) {
		ArrayList<String> classifications = new ArrayList<String>();
		for(int i = 0 ; i < testing.size() ; i++) {
			FImage img = testing.get(i);
			List<IntDoublePair> result = knn.searchKNN(getTinyImageFeature(img), k);
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
			String outputRes = testing.getID(i) + " ";
			outputRes += finalRes;
			classifications.add(outputRes);
		}
		return classifications;
	}
	
	public static void classificationRun(){
		try {
			GroupedDataset<String, VFSListDataset<FImage>, FImage> train =new VFSGroupDataset<FImage>("/Users/shintaroonuma/Documents/cwImages/training", ImageUtilities.FIMAGE_READER);
			VFSListDataset<FImage> test = new VFSListDataset<FImage>("/Users/shintaroonuma/Documents/cwImages/testing", ImageUtilities.FIMAGE_READER);
    		//GroupedDataset<String, ListDataset<FImage>, FImage> trainNewFormat = splitter.getTrainingDataset();
			DisplayUtilities.display(test.getID(0),test.get(0));
    		VFStrainKNN(train);
    		ArrayList<String> ans = classKNN(test,1);
    		for(String s : ans) {
    			System.out.println(s);
    		}
    	} catch (FileSystemException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	

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
			
			for(int i = 0 ; i < iter ; i++) {
				GroupedRandomSplitter<String,FImage> splitter = new GroupedRandomSplitter(input,90,0,10);
	    		GroupedDataset<String, ListDataset<FImage>, FImage> train = splitter.getTrainingDataset();
	    		GroupedDataset<String, ListDataset<FImage>, FImage> test = splitter.getTestDataset();
	    		trainKNN(train);
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
			
			System.out.println("k = 1: " + a1);
			System.out.println("k = 2: " + a2);
			System.out.println("k = 3: " + a3);
			System.out.println("k = 4: " + a4);
			System.out.println("k = 5: " + a5);
			System.out.println("k = 6: " + a6);
			System.out.println("k = 7: " + a3);
			System.out.println("k = 8: " + a4);
			System.out.println("k = 9: " + a5);
			System.out.println("k = 10: " + a6);
			
			
    	} catch (FileSystemException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
    public static void main( String[] args ) {
    	classificationRun();
    	//accuracyRun();
    }
}
