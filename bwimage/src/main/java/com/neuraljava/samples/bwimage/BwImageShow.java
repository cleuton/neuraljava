package com.neuraljava.samples.bwimage;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;

public class BwImageShow {

	/**
	 * Mostra a composição de uma imagem monocromática
	 * @param args String - Path da imagem
	 * @throws IOException 
	 */
	public static void main(String [] args) throws IOException {
		BufferedImage image = ImageIO.read(new File(args[0]));
		drawImage(image);
		int[][] pixels = new int[image.getWidth()][];

		for (int x = 0; x < image.getWidth(); x++) {
		    pixels[x] = new int[image.getHeight()];

		    for (int y = 0; y < image.getHeight(); y++) {
		        pixels[x][y] = (int)(image.getRGB(y, x) & 0xFF);
		    }
		    System.out.println(Arrays.toString(pixels[x]));
		}
	}
	

	private static void drawImage(BufferedImage image) {
		JLabel picLabel = new JLabel(new ImageIcon(image));
		JOptionPane.showMessageDialog(null, picLabel, "Imagem", JOptionPane.PLAIN_MESSAGE, null);
	}
	
}
