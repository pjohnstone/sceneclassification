package uk.ac.soton.ecs.comp3204.group5;

import org.openimaj.data.identity.Identifiable;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;

/**
 * Implementation of the Caltech101 Record class for scene classification
 */
public class Record implements Identifiable, ImageProvider<FImage> {

    private final FImage image;
    private final String id;
    private final String getImageObjectClass;

    public Record(FImage image, String id, String imageObjectClass) {
        this.image = image;
        this.id = id;
        this.getImageObjectClass = imageObjectClass;
    }

    @Override
    public FImage getImage() { return this.image; }

    @Override
    public String getID() {
        return this.id;
    }

    public String getImageObjectClass() {
        return this.getImageObjectClass;
    }

}