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

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class TrainModel {
	private static final Logger log = Logger.getLogger(TrainModel.class.getName());
	public static void train(String imagePath, String modelPath, int numLabels) throws IOException {
		
		// Random
		int seed = 42;
		Random random = new Random(seed);
		
		// Hiperparametros
		
		int height = 80;
		int width = 80;
		int channels = 1;
		int batchSize = 10;
		double trainTestRatio = 0.6;
		int iterations = 1;
		int epochs = 500;
		double learningRate = 0.01;
		
		// Preparar dataset com o nosso label generator:
		
        LabelGen labelMaker = new LabelGen();
        File mainPath = new File(imagePath);
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, random);
        int numExamples = Math.toIntExact(fileSplit.length());
        BalancedPathFilter pathFilter = new BalancedPathFilter(random, labelMaker, numExamples, numLabels, 0);
        
        // Separação treino e teste:
        
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, trainTestRatio, 1 - trainTestRatio);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];
        
        // Configuração da rede:
        
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
        
        // Carga dos dados:
        
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        
        // Iterador de treino:
        
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        DataSetIterator trainIter;
        trainRR.initialize(trainData, null);
        trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numLabels);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        
        // Iterador de teste:
        
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numLabels);
        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);
        
        // Configuração de Early Stopping:
        
        int maxEpochsWithNoImprovement = 10;
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
        		.epochTerminationConditions(new MaxEpochsTerminationCondition(epochs),new ScoreImprovementEpochTerminationCondition(maxEpochsWithNoImprovement) )
        		.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
        		.scoreCalculator(new DataSetLossCalculator(testIter, true))
                .evaluateEveryNEpochs(1)
        		.modelSaver(new LocalFileModelSaver(modelPath))
        		.build();
        
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,trainIter);

        // Treinamento da rede: 

        EarlyStoppingResult result = trainer.fit();
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());
        
        // Recupera o modelo com melhor desempenho:
        
        MultiLayerNetwork net = (MultiLayerNetwork) result.getBestModel();  
        
        // Predições:

        trainIter.reset();
        DataSet testDataSet = trainIter.next();
        List<String> allClassLabels = trainRR.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = net.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        System.out.print("\nNome da pessoa: " + expectedResult + " predição: " + modelPrediction + "\n\n");
        
        
	}
	
}
