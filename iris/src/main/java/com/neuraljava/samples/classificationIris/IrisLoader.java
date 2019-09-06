package com.neuraljava.samples.classificationIris;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class IrisLoader {
	public static void main (String [] args) throws IOException {
		
		Nd4j.getRandom().setSeed(124);
		MultiLayerNetwork net = MultiLayerNetwork.load(new File("/tmp/irisModelSave.zip"), true);
		
        // Predição: 
        // Valor esperado: 1 Iris-Versicolor
        double [][] valores = new double[][] {{5.61, 2.88, 4.0, 1.1}};
        INDArray vetor = Nd4j.create(valores);
        System.out.println("Esperado: 1, previsto: " + net.output(vetor));
	}
}
