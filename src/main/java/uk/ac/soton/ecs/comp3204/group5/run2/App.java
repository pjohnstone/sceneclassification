package uk.ac.soton.ecs.comp3204.group5.run2;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.util.Map;

/**
 * OpenIMAJ Hello world!
 *
 */
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

        
    }
}
