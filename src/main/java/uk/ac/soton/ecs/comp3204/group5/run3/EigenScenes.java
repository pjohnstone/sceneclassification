package uk.ac.soton.ecs.comp3204.group5.run3;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.DoubleFV2FImage;
import org.openimaj.image.feature.FImage2DoubleFV;
import org.openimaj.io.IOUtils;
import org.openimaj.io.ReadWriteableBinary;
import org.openimaj.math.matrix.algorithm.pca.ThinSvdPrincipalComponentAnalysis;
import org.openimaj.ml.pca.FeatureVectorPCA;
import org.openimaj.ml.training.BatchTrainer;
import org.openimaj.util.array.ArrayUtils;
import uk.ac.soton.ecs.comp3204.group5.Record;

public class EigenScenes implements BatchTrainer<FImage>, FeatureExtractor<DoubleFV, FImage>, ReadWriteableBinary {
    private FeatureVectorPCA pca;
    private int width;
    private int height;
    private int numComponents;
    FeatureExtractor<DoubleFV, Record> extractor;

    /**
     * For serialisation
     */
    protected EigenScenes() {
    }

    /**
     * Construct with the given number of principal components.
     *
     * @param numComponents
     *            the number of PCs
     */
    public EigenScenes(int numComponents, FeatureExtractor<DoubleFV, Record> extractor) {
        this.numComponents = numComponents;
        this.extractor = extractor;
        pca = new FeatureVectorPCA(new ThinSvdPrincipalComponentAnalysis(numComponents));
    }

    @Override
    public DoubleFV extractFeature(FImage img) {
        final DoubleFV feature = FImage2DoubleFV.INSTANCE.extractFeature(img);

        return pca.project(feature);
    }

    public DoubleFV extractFeature(Record record) {
        final DoubleFV feature = extractor.extractFeature(record);

        return pca.project(feature);
    }

    public void train(List<? extends FImage> data) {
        final double[][] features = new double[data.size()][];

        width = data.get(0).width;
        height = data.get(0).height;

        for (int i = 0; i < features.length; i++)
            features[i] = FImage2DoubleFV.INSTANCE.extractFeature(data.get(i)).values;

        pca.learnBasis(features);
    }

    public void train(ArrayList<Record> data) {
        final double[][] features = new double[data.size()][];

        FImage image = data.get(0).getImage();

        width = image.width;
        height = image.height;

        for (int i = 0; i < features.length; i++)
            features[i] = extractor.extractFeature(data.get(i)).values;

        pca.learnBasis(features);
    }

    /**
     * Reconstruct an image from a weight vector
     *
     * @param weights
     *            the weight vector
     * @return the reconstructed image
     */
    public FImage reconstruct(DoubleFV weights) {
        return DoubleFV2FImage.extractFeature(pca.generate(weights), width, height);
    }

    /**
     * Reconstruct an image from a weight vector
     *
     * @param weights
     *            the weight vector
     * @return the reconstructed image
     */
    public FImage reconstruct(double[] weights) {
        return new FImage(ArrayUtils.reshapeFloat(pca.generate(weights), width, height));
    }

    /**
     * Draw a principal component as an image. The image will be normalised so
     * it can be displayed correctly.
     *
     * @param pc
     *            the index of the PC to draw.
     * @return an image showing the PC.
     */
    public FImage visualisePC(int pc) {
        return new FImage(ArrayUtils.reshapeFloat(pca.getPrincipalComponent(pc), width, height)).normalise();
    }

    @Override
    public void readBinary(DataInput in) throws IOException {
        width = in.readInt();
        height = in.readInt();
        numComponents = in.readInt();
        pca = IOUtils.read(in);
    }

    @Override
    public byte[] binaryHeader() {
        return "EigI".getBytes();
    }

    @Override
    public void writeBinary(DataOutput out) throws IOException {
        out.writeInt(width);
        out.writeInt(height);
        out.writeInt(numComponents);
        IOUtils.write(pca, out);
    }

    @Override
    public String toString() {
        return String.format("EigenImages[width=%d; height=%d; pca=%s]", width, height, pca);
    }

    /**
     * Get the number of PCA components selected by this {@link org.openimaj.image.model.EigenImages}
     * object.
     *
     * @return the number of PCA components.
     */
    public int getNumComponents() {
        return numComponents;
    }
}