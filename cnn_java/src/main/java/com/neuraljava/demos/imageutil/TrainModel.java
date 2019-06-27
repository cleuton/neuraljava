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
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import javax.imageio.ImageIO;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TrainModel {
	private static final Logger log = Logger.getLogger(TrainModel.class.getName());
	public static void train(String imagePath, String modelPath, int numLabels, String testImage) throws IOException {
		
		// Random
		int seed = 42;
		Random random = new Random(seed);
		
		// Hiperparameters
		int height = 80;
		int width = 80;
		int channels = 1;
		int batchSize = 10;
		double trainTestRatio = 0.7;
		int iterations = 1;
		int epochs = 50;
		double learningRate = 0.01;
		
		// Prepare dataset:
		
        LabelGen labelMaker = new LabelGen();
        File mainPath = new File(imagePath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, random);
        int numExamples = Math.toIntExact(fileSplit.length());
        
        // Network configuration:
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1,1)
                        .nOut(32)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height,width,1)) 
                .build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);

        network.init();
        
        // Initialize the user interface backend
        // Be careful: It slows the performance for classification!
        /*
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();         
        uiServer.attach(statsStorage);
        int listenerFrequency = 1;
        network.setListeners(new StatsListener(statsStorage,listenerFrequency));
        */
        
        
        // Load data:
        
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(fileSplit);
        
        //DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
        DataSetIterator dataIter = new RecordReaderDataSetIterator.Builder(recordReader, batchSize)
        	       .classification(1, numLabels)
        	       .preProcessor(new ImagePreProcessingScaler())      
        	       .build();
  
        network.fit(dataIter, epochs);
        
        List<String> allClassLabels = recordReader.getLabels();
        System.out.println("Labels in order: " + allClassLabels);
        
        dataIter.reset();
        DataSet testDataSet = dataIter.next();

        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = network.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelPrediction + "\n\n");
        
        
        ModelSerializer.writeModel(network, modelPath + "/model.dat", true);
        
        // Load test data:
 		BufferedImage imagem = ImageIO.read(new File(testImage));
 		String label = labelMaker.getLabelForPath(testImage).toString();
 		byte[] pixels = ((DataBufferByte) imagem.getRaster().getDataBuffer()).getData();
 		double [] vetorImagem = new double[pixels.length];
 		for (int x=0;x<pixels.length; x++) {
 			vetorImagem[x] = pixels[x];
 		}
 		int [] shape = {1,1,height,width};
 		INDArray entrada = Nd4j.create(vetorImagem,shape,'c'); 
 		
 		// Roda inferências:
 		entrada = entrada.div(255);
 		INDArray results = network.output(entrada);
 		System.out.println("Label: " + label + " resultados: " + results);
        
        //uiServer.stop();  // Always remember to do that!
	}
	
    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }


    private static SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }
}
