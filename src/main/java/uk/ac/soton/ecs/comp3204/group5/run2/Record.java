package uk.ac.soton.ecs.comp3204.group5.run2;

import org.openimaj.data.identity.Identifiable;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageProvider;

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