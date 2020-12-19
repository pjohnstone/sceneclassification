package uk.ac.soton.ecs.comp3204.group5.run2;

import org.openimaj.data.identity.Identifiable;
import org.openimaj.image.Image;
import org.openimaj.image.ImageProvider;

public class Record<IMAGE extends Image<?, IMAGE>> implements Identifiable, ImageProvider<IMAGE> {

    private String id;
    private IMAGE image;
    private String getImageObjectClass;

    public Record(String id, IMAGE image, String imageObjectClass) {
        this.id = id;
        this.image = image;
        this.getImageObjectClass = imageObjectClass;
    }

    @Override
    public String getID() {
        return this.id;
    }

    @Override
    public IMAGE getImage() {
        return this.image;
    }

    public String getImageObjectClass() {
        return this.getImageObjectClass;
    }
}