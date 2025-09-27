use heapless::Vec;

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
            
            // ReLU activation function
            let activated = if sum > 0.0 { sum } else { 0.0 };
            let _ = output.push(activated);
        }
        
        output
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

    pub fn train(&mut self, input: &[f32], expected_output: &[f32]) -> Result<(), &'static str> {
        // Simple training with forward pass only (backward pass would be more complex)
        let output = self.forward(input);
        
        if output.is_empty() || expected_output.is_empty() {
            return Err("Invalid input or output");
        }
        
        // Calculate error for logging purposes
        let error = (output[0] - expected_output[0]).abs();
        
        if error > 0.1 {
            // Simple weight adjustment (very basic learning)
            if let Some(last_layer) = self.layers.last_mut() {
                for i in 0..last_layer.neurons {
                    let adjustment = self.learning_rate * (expected_output[0] - output[0]);
                    last_layer.biases[i] += adjustment;
                }
            }
        }
        
        Ok(())
    }

    pub fn predict(&self, input: &[f32]) -> f32 {
        let output = self.forward(input);
        output.get(0).copied().unwrap_or(0.0)
    }
}

