package com.neuraljava.demos.gpu;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Random;

import org.apache.log4j.BasicConfigurator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

public class MatMul {

	
	
	private static float [][] iterativo(float[][] A, float [][] B) {

		float [][] AB = new float[A.length][B[0].length];
		int linAB=0;
		int colAB=0;
		for (int linA=0; linA<A.length; linA++) {
			for (int colB=0; colB<B[0].length; colB++) {
				for (int colA=0; colA<A[0].length; colA++) {
					AB[linAB][colAB] += A[linA][colA] * B[colA][colB];
				}
				colAB++;
				if(colAB >= AB[0].length) {
					colAB = 0;
					linAB++;
				}
			}
		}

		return AB;
	}
	
	
	public static void main (String [] args) {
		
		int seed = 42;
		int linhasA = 500;
		int colunasA = 300;
		int linhasB = 300;
		int colunasB = 400;
		int linhasC = linhasA;
		int colunasC = colunasB;
		Random random = new Random(seed);
		BasicConfigurator.configure();
		
		float [][] A = new float[linhasA][colunasA];
		for(int x=0;x<linhasA;x++){
	        float[] vetor = A[x];
	        Arrays.fill (vetor, new Random().nextFloat());
		}
		
		float [][] B = new float[linhasB][colunasB];
		for(int x=0;x<linhasB;x++){
	        float[] vetor = B[x];
	        Arrays.fill (vetor, new Random().nextFloat());
		}
		
		int iteracoes = 1;
		
		float [][] AB = null;
		long startTime = System.nanoTime();
		AB = iterativo(A,B);
		
		long endTime = System.nanoTime();
		long cpuTime = endTime - startTime;
		System.out.println("CPU Interativo \t(nanossegundos): " + NumberFormat.getNumberInstance().format(cpuTime));
		//printMatriz(AB);
		
		
		// Agora, em paralelo usando a GPU
		INDArray ndA = Nd4j.create(A);
		INDArray ndB = Nd4j.create(B);
		INDArray ndC  = null;
		startTime = System.nanoTime();
		ndC = ndA.mmul(ndB);
		endTime = System.nanoTime();
		long gpuTime = endTime - startTime;
		System.out.println("\n\nGPU Paralelo \t(nanossegundos): " + NumberFormat.getNumberInstance().format(gpuTime));
		//printMatriz(ndC.toFloatMatrix());
		
		System.out.println("\n\nPercentual de tempo no cálculo da GPU com relação à CPU: " + (((double)gpuTime / (double) cpuTime))*100);
	
	}

	private static void printMatriz(float[][] matriz) {
		DecimalFormat df = new DecimalFormat("0.##");
		for (int lin=0;lin<matriz.length;lin++) {
			System.out.println(" ");
			for (int col=0; col<matriz[0].length;col++) {
				System.out.print("\t" + df.format(matriz[lin][col]));
			}
		}
		System.out.println(" ");
 	}
}
