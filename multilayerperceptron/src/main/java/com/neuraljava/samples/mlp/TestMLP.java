package com.neuraljava.samples.mlp;

import java.util.Arrays;
import java.util.Random;

public class TestMLP {

	public static void main(String[] args) throws Exception {
		MLP mlp = new MLP();
		double [][] X = {
				{1,0},{0,1},{1,1},{0,0}
		};
		double [] y = {1,1,0,0};
		mlp.learningHate = 0.4;
		mlp.iterations = 500;
		mlp.treinar(X, y);
		double [][] teste = {{0,1},{0,0},{1,0},{1,1}};
		double [] real = {1,0,1,0};
		for(int i=0;i<4;i++) {
			mlp.forwardPropagation(teste[i]);
			System.out.println("Teste " + i 
					+ " previsto: " + real[i]
					+ " calculado: " + mlp.outputValue);
		}
	}

}
