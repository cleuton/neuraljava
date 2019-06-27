/*
 Copyright 2019 Cleuton Sampaio

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and limitations under the License.

There is also the complete Apache License v2 available in HTML and TXT formats.
 */
package com.neuraljava.demos.imageutil;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.face.alignment.RotateScaleAligner;
import org.openimaj.image.processing.face.detection.keypoints.FKEFaceDetector;
import org.openimaj.image.processing.face.detection.keypoints.KEDetectedFace;

/*
This example uses Labeled Faces in the Wild - LFW database: 
http://vis-www.cs.umass.edu/lfw/

Each image is preprocessed to align the faces, convert to B&W and crop to 80 x 80 pixels.
This is done by OpenIMAJ package. 
 */
public class PrepareFaces {
	public static List<BufferedImage> getFaces(String imagePath) throws IOException {
		List<BufferedImage> facesList = new ArrayList<BufferedImage>();
		File arq = new File(imagePath);
		BufferedImage imagem = ImageIO.read(arq);
		FImage fimage = ImageUtilities.createFImage(imagem);
		FKEFaceDetector detector = new FKEFaceDetector(1);
		List<KEDetectedFace> faces = detector.detectFaces(fimage);	
		RotateScaleAligner aligner = new RotateScaleAligner();
		for (KEDetectedFace face : faces) {
			FImage aligned = aligner.align(face);
			BufferedImage bfi = ImageUtilities.createBufferedImage(aligned);
			facesList.add(bfi);
		}
		return facesList;
	}
	
	public static Set<String> prepareImages(String sourcePath, String targetPath) throws IOException {
		Set<String> names = new HashSet<String>();
		File[] files = new File(sourcePath).listFiles();
	    for (File file : files) {
	    	String name = getNameFromFile(file);
	    	names.add(name);
	    	List<BufferedImage> faces = getFaces(file.getAbsolutePath());
	    	int faceCount = 0;
	    	for (BufferedImage img : faces) {
	    		int pos = file.getName().indexOf('.');
	    		String fileName = file.getName().substring(0,pos) + faceCount + ".jpg";
	    		File outputfile = new File(targetPath + "/" +  fileName);
	    		ImageIO.write(img, "jpg", outputfile);
	    		faceCount++;
	    	}
	        System.out.println("Name: " + name);
	    }
	    return names;
	}

	private static String getNameFromFile(File file) {
		int dot = file.getName().lastIndexOf("_");
		return file.getName().substring(0, dot);
	}		
	
}
