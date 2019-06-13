package com.neuraljava.demos.kerasmodel;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.MnistImageFile;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

public class RunModel {

	public static void main(String[] args) throws IOException, InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
		
		// Carrega o modelo pré-treinado: 
		String txtModel = new ClassPathResource("meu_mnist.h5").getFile().getPath();
		MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(txtModel);
		System.out.println("OK: " + model.getnLayers());
		
		// Carrega um arquivo de imagem tipo MNIST:
		BufferedImage imagem = ImageIO.read(new File(args[0]));
		byte[] pixels = ((DataBufferByte) imagem.getRaster().getDataBuffer()).getData();
		double [] vetorImagem = new double[pixels.length];
		for (int x=0;x<pixels.length; x++) {
			vetorImagem[x] = pixels[x];
		}
		int [] shape = {1,1,28,28};
		INDArray entrada = Nd4j.create(vetorImagem,shape,'c'); 
		
		// Roda inferências:
		INDArray results = model.output(entrada);
		System.out.println(results);
		
	}

}
