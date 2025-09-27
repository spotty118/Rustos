use heapless::Vec;
use micromath;

const MAX_LAYERS: usize = 3;
const MAX_NEURONS_PER_LAYER: usize = 16;
const INPUT_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 1;

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: [[f32; MAX_NEURONS_PER_LAYER]; MAX_NEURONS_PER_LAYER],
    pub biases: [f32; MAX_NEURONS_PER_LAYER],
    pub neurons: usize,
}

impl Layer {
    pub fn new(neurons: usize) -> Self {
        Self {
            weights: [[0.0; MAX_NEURONS_PER_LAYER]; MAX_NEURONS_PER_LAYER],
            biases: [0.0; MAX_NEURONS_PER_LAYER],
            neurons,
        }
    }

    pub fn initialize_random(&mut self) {
        // Simple pseudo-random initialization using a linear congruential generator
        let mut seed = 1234u32;
        
        for i in 0..self.neurons {
            for j in 0..MAX_NEURONS_PER_LAYER {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                self.weights[i][j] = ((seed as i32) as f32) / (i32::MAX as f32) * 0.1;
            }
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            self.biases[i] = ((seed as i32) as f32) / (i32::MAX as f32) * 0.1;
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32, MAX_NEURONS_PER_LAYER> {
        let mut output = Vec::new();
        
        for i in 0..self.neurons {
            let mut sum = self.biases[i];
            
            for j in 0..input.len().min(MAX_NEURONS_PER_LAYER) {
                sum += input[j] * self.weights[i][j];
            }
            
            // Sigmoid activation function for better gradient flow
            let activated = 1.0 / (1.0 + micromath::F32Ext::exp(-sum));
            let _ = output.push(activated);
        }
        
        output
    }
    
    /// Compute layer output with activation gradients for backpropagation
    pub fn forward_with_gradients(&self, input: &[f32]) -> (Vec<f32, MAX_NEURONS_PER_LAYER>, Vec<f32, MAX_NEURONS_PER_LAYER>) {
        let mut output = Vec::new();
        let mut gradients = Vec::new();
        
        for i in 0..self.neurons {
            let mut sum = self.biases[i];
            
            for j in 0..input.len().min(MAX_NEURONS_PER_LAYER) {
                sum += input[j] * self.weights[i][j];
            }
            
            // Sigmoid activation
            let activated = 1.0 / (1.0 + micromath::F32Ext::exp(-sum));
            // Sigmoid derivative for backpropagation
            let gradient = activated * (1.0 - activated);
            
            let _ = output.push(activated);
            let _ = gradients.push(gradient);
        }
        
        (output, gradients)
    }
    
    /// Update layer weights using backpropagation
    pub fn update_weights(&mut self, input: &[f32], error_gradients: &[f32], learning_rate: f32) {
        for i in 0..self.neurons {
            if i < error_gradients.len() {
                let error_grad = error_gradients[i];
                
                // Update weights
                for j in 0..input.len().min(MAX_NEURONS_PER_LAYER) {
                    self.weights[i][j] += learning_rate * error_grad * input[j];
                }
                
                // Update bias
                self.biases[i] += learning_rate * error_grad;
            }
        }
    }
}

pub struct NeuralNetwork {
    layers: Vec<Layer, MAX_LAYERS>,
    learning_rate: f32,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            learning_rate: 0.01,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        // Create a simple 3-layer network: input -> hidden -> output
        let mut input_layer = Layer::new(8);
        input_layer.initialize_random();
        
        let mut hidden_layer = Layer::new(4);
        hidden_layer.initialize_random();
        
        let mut output_layer = Layer::new(1);
        output_layer.initialize_random();
        
        self.layers.clear();
        self.layers.push(input_layer).map_err(|_| "Failed to add input layer")?;
        self.layers.push(hidden_layer).map_err(|_| "Failed to add hidden layer")?;
        self.layers.push(output_layer).map_err(|_| "Failed to add output layer")?;
        
        Ok(())
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32, MAX_NEURONS_PER_LAYER> {
        let mut current_input = Vec::new();
        
        // Convert input slice to Vec for processing
        for &val in input.iter().take(MAX_NEURONS_PER_LAYER) {
            let _ = current_input.push(val);
        }
        
        for layer in &self.layers {
            let output = layer.forward(&current_input);
            current_input = output;
        }
        
        current_input
    }

    pub fn train(&mut self, input: &[f32], expected_output: &[f32]) -> Result<f32, &'static str> {
        if input.is_empty() || expected_output.is_empty() {
            return Err("Invalid input or output");
        }
        
        // Forward pass storing intermediate values
        let mut layer_outputs = Vec::<Vec<f32, MAX_NEURONS_PER_LAYER>, MAX_LAYERS>::new();
        let mut layer_gradients = Vec::<Vec<f32, MAX_NEURONS_PER_LAYER>, MAX_LAYERS>::new();
        
        let mut current_input = Vec::new();
        for &val in input.iter().take(MAX_NEURONS_PER_LAYER) {
            let _ = current_input.push(val);
        }
        
        // Forward pass through all layers
        for layer in &self.layers {
            let (output, gradients) = layer.forward_with_gradients(&current_input);
            current_input = output.clone();
            let _ = layer_outputs.push(output);
            let _ = layer_gradients.push(gradients);
        }
        
        if layer_outputs.is_empty() {
            return Err("No layer outputs");
        }
        
        // Calculate output error
        let final_output = &layer_outputs[layer_outputs.len() - 1];
        if final_output.is_empty() {
            return Err("Empty final output");
        }
        
        let error = final_output[0] - expected_output[0];
        let mean_squared_error = error * error;
        
        // Backward pass (simplified backpropagation)
        let mut error_gradients = Vec::<f32, MAX_NEURONS_PER_LAYER>::new();
        let _ = error_gradients.push(error * layer_gradients[layer_gradients.len() - 1][0]);
        
        // Update weights from output to input
        for layer_idx in (0..self.layers.len()).rev() {
            let layer_input = if layer_idx == 0 {
                // First layer uses original input
                let mut input_vec = Vec::new();
                for &val in input.iter().take(MAX_NEURONS_PER_LAYER) {
                    let _ = input_vec.push(val);
                }
                input_vec
            } else {
                // Other layers use previous layer output
                layer_outputs[layer_idx - 1].clone()
            };
            
            self.layers[layer_idx].update_weights(&layer_input, &error_gradients, self.learning_rate);
            
            // Propagate error to previous layer (simplified)
            if layer_idx > 0 {
                let mut new_error_gradients = Vec::new();
                for i in 0..self.layers[layer_idx - 1].neurons {
                    let mut error_sum = 0.0;
                    for j in 0..error_gradients.len() {
                        error_sum += error_gradients[j] * self.layers[layer_idx].weights[j][i];
                    }
                    let _ = new_error_gradients.push(error_sum * layer_gradients[layer_idx - 1][i]);
                }
                error_gradients = new_error_gradients;
            }
        }
        
        Ok(mean_squared_error)
    }
    
    /// Train the network with multiple epochs
    pub fn train_epochs(&mut self, training_data: &[(&[f32], &[f32])], epochs: usize) -> Result<f32, &'static str> {
        let mut total_error = 0.0;
        let mut sample_count = 0;
        
        for epoch in 0..epochs {
            let mut epoch_error = 0.0;
            
            for (input, expected) in training_data {
                match self.train(input, expected) {
                    Ok(error) => {
                        epoch_error += error;
                        sample_count += 1;
                    }
                    Err(e) => {
                        crate::println!("[NN] Training error at epoch {}: {}", epoch, e);
                    }
                }
            }
            
            total_error += epoch_error;
            
            // Adaptive learning rate
            if epoch % 10 == 0 && epoch > 0 {
                let avg_error = epoch_error / training_data.len() as f32;
                if avg_error > 0.1 {
                    self.learning_rate *= 1.05; // Increase if error is high
                } else if avg_error < 0.01 {
                    self.learning_rate *= 0.95; // Decrease if error is low
                }
                
                // Keep learning rate in reasonable bounds
                self.learning_rate = self.learning_rate.max(0.001).min(0.1);
            }
        }
        
        if sample_count > 0 {
            Ok(total_error / sample_count as f32)
        } else {
            Err("No training samples processed")
        }
    }

    pub fn predict(&self, input: &[f32]) -> f32 {
        let output = self.forward(input);
        output.get(0).copied().unwrap_or(0.0)
    }
}

#[test_case]
fn test_neural_network_creation() {
    let mut nn = NeuralNetwork::new();
    assert!(nn.initialize().is_ok());
    assert_eq!(nn.layers.len(), 3);
}

#[test_case]
fn test_neural_network_forward_pass() {
    let mut nn = NeuralNetwork::new();
    let _ = nn.initialize();
    
    let input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 0.0];
    let output = nn.forward(&input);
    
    assert!(!output.is_empty());
}