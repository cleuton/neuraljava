package com.neuraljava.samples.mlpgen.api;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/*

Copyright 2018 Cleuton Sampaio

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 

Este trabalho é para demonstração de redes neurais e não tem objetivo 
de desempenho ou precisão. O Autor não se responsabiliza pelo seu uso e
não fornecerá suporte. 
*/
public class Layer {
	public Activation activation;
	public List<Node> nodes;
	public int number;
	Node bias;
	
	public Layer(int numNodes, Activation activation, Model model) {
		super();
		this.activation = activation;
		this.nodes = new ArrayList<Node>();
		this.bias = new Node();
		this.bias.sinapses = new ArrayList<Sinapse>();
		this.number = model.layers.size() + 1;
		this.bias.layerNumber = this.number;
		for (int x=0; x<numNodes; x++) {
			Node node = new Node();
			node.sinapses = new ArrayList<Sinapse>();
			this.nodes.add(node);
			node.layerNumber = this.number;
			node.nodeNumber = x+1;
		}
		// Criamos as sinapses da camada anterior, conectando esta camada à ela.
		if (model.layers.size() > 0) {
			// A input layer não tem camada anterior
			Layer previous =  model.layers.get(model.layers.size()-1); // Pega a última inserida
			for (Node nprev : previous.nodes) {
				for (Node natu : this.nodes) {
					Sinapse sinapse = new Sinapse();
					sinapse.finalNode = natu;
					sinapse.weight = model.getRandom();
					nprev.sinapses.add(sinapse);
				}
			}
			// Bias da camada anterior (um pouco de repetição, mas dá para entender bem)
			for (Node natu : this.nodes) {
				Sinapse sinapse = new Sinapse();
				sinapse.finalNode = natu;
				sinapse.weight = model.getRandom();
				previous.bias.sinapses.add(sinapse);
			}	
		}
		else {
			model.firstLayer = this;
		}
		model.lastLayer = this;
	}

	@Override
	public boolean equals(Object obj) {
		return ((Layer)obj).number == this.number;
	}

	@Override
	public String toString() {
		String saida = "\n[Layer. Number : " + this.number
					 + "\nBias: " 
					 + this.bias
				     + "\nnodes:\n"
				     + this.nodes
				     + "\n]";
		return saida;
	}
	
	
}
