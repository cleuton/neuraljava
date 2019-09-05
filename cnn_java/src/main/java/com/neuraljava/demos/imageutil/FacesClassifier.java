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
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.face.alignment.RotateScaleAligner;
import org.openimaj.image.processing.face.detection.keypoints.FKEFaceDetector;
import org.openimaj.image.processing.face.detection.keypoints.KEDetectedFace;

public class FacesClassifier {

	public static void main (String [] args) throws IOException {
		/*
		 * args 0: raw images path
		 * args 1: processed images path
		 * args 2: model save path
		 */
		Set<String> classes = PrepareFaces.prepareImages(args[0], args[1]);
		System.out.println("Classes: " + classes.size());
		TrainModel.train(args[1], args[2], classes.size());
	}
}
