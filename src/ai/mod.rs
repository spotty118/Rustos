use heapless::Vec;
use spin::Mutex;
use lazy_static::lazy_static;

pub mod neural_network;
pub mod inference_engine;
pub mod learning;

use neural_network::NeuralNetwork;
use inference_engine::InferenceEngine;

const MAX_INPUT_SIZE: usize = 64;
const MAX_LEARNED_PATTERNS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AIStatus {
    Initializing,
    Ready,
    Learning,
    Inferencing,
    Error,
}

impl core::fmt::Display for AIStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            AIStatus::Initializing => write!(f, "Initializing"),
            AIStatus::Ready => write!(f, "Ready"),
            AIStatus::Learning => write!(f, "Learning"),
            AIStatus::Inferencing => write!(f, "Inferencing"),
            AIStatus::Error => write!(f, "Error"),
        }
    }
}

pub struct AISystem {
    status: AIStatus,
    neural_network: NeuralNetwork,
    inference_engine: InferenceEngine,
    learned_patterns: Vec<[f32; MAX_INPUT_SIZE], MAX_LEARNED_PATTERNS>,
    input_buffer: Vec<u8, MAX_INPUT_SIZE>,
}

impl AISystem {
    pub fn new() -> Self {
        Self {
            status: AIStatus::Initializing,
            neural_network: NeuralNetwork::new(),
            inference_engine: InferenceEngine::new(),
            learned_patterns: Vec::new(),
            input_buffer: Vec::new(),
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        crate::println!("[AI] Initializing neural network...");
        self.neural_network.initialize()?;
        
        crate::println!("[AI] Initializing inference engine...");
        self.inference_engine.initialize()?;
        
        crate::println!("[AI] Loading pre-trained patterns...");
        self.load_default_patterns()?;
        
        self.status = AIStatus::Ready;
        crate::println!("[AI] AI system successfully initialized!");
        Ok(())
    }

    pub fn get_status(&self) -> AIStatus {
        self.status
    }

    pub fn process_keyboard_input(&mut self, character: char) {
        if self.input_buffer.len() < MAX_INPUT_SIZE {
            let _ = self.input_buffer.push(character as u8);
            
            // Trigger learning when we have enough data
            if self.input_buffer.len() >= 8 {
                self.learn_from_input();
            }
        }
    }

    pub fn periodic_task(&mut self) {
        // Perform background AI tasks like pattern recognition
        if self.status == AIStatus::Ready && !self.input_buffer.is_empty() {
            self.status = AIStatus::Inferencing;
            let _ = self.run_inference();
            self.status = AIStatus::Ready;
        }
    }

    fn learn_from_input(&mut self) {
        self.status = AIStatus::Learning;
        
        // Convert input buffer to neural network input
        let mut pattern = [0.0f32; MAX_INPUT_SIZE];
        for (i, &byte) in self.input_buffer.iter().enumerate() {
            if i < MAX_INPUT_SIZE {
                pattern[i] = (byte as f32) / 255.0; // Normalize to 0-1 range
            }
        }
        
        // Store the learned pattern
        if self.learned_patterns.len() < MAX_LEARNED_PATTERNS {
            let _ = self.learned_patterns.push(pattern);
            crate::println!("[AI] Learned new pattern from input: {} patterns stored", 
                           self.learned_patterns.len());
        }
        
        // Train the neural network
        let _ = self.neural_network.train(&pattern, &[1.0]); // Simple binary classification
        
        // Clear input buffer
        self.input_buffer.clear();
    }

    fn run_inference(&mut self) -> Result<f32, &'static str> {
        if self.input_buffer.is_empty() {
            return Ok(0.0);
        }

        // Convert current input to neural network format
        let mut input = [0.0f32; MAX_INPUT_SIZE];
        for (i, &byte) in self.input_buffer.iter().enumerate() {
            if i < MAX_INPUT_SIZE {
                input[i] = (byte as f32) / 255.0;
            }
        }

        // Run inference
        let result = self.inference_engine.infer(&input)?;
        
        if result > 0.7 {
            crate::println!("[AI] High confidence pattern detected: {:.2}", result);
        }
        
        Ok(result)
    }

    fn load_default_patterns(&mut self) -> Result<(), &'static str> {
        // Load some default patterns for demonstration
        let mut pattern1 = [0.0f32; MAX_INPUT_SIZE];
        pattern1[0] = 0.1; pattern1[1] = 0.2; pattern1[2] = 0.3; pattern1[3] = 0.4; pattern1[4] = 0.5;
        
        let mut pattern2 = [0.0f32; MAX_INPUT_SIZE];
        pattern2[0] = 0.5; pattern2[1] = 0.4; pattern2[2] = 0.3; pattern2[3] = 0.2; pattern2[4] = 0.1;
        
        let _ = self.learned_patterns.push(pattern1);
        let _ = self.learned_patterns.push(pattern2);
        
        Ok(())
    }
}

lazy_static! {
    static ref AI_SYSTEM: Mutex<AISystem> = Mutex::new(AISystem::new());
}

pub fn init_ai_system() {
    let mut ai = AI_SYSTEM.lock();
    match ai.initialize() {
        Ok(_) => crate::println!("[AI] AI system initialization completed successfully"),
        Err(e) => crate::println!("[AI] AI system initialization failed: {}", e),
    }
}

pub fn get_ai_status() -> AIStatus {
    AI_SYSTEM.lock().get_status()
}

pub fn process_keyboard_input(character: char) {
    AI_SYSTEM.lock().process_keyboard_input(character);
}

pub fn periodic_ai_task() {
    AI_SYSTEM.lock().periodic_task();
}

#[test_case]
fn test_ai_initialization() {
    let mut ai = AISystem::new();
    assert!(ai.initialize().is_ok());
    assert_eq!(ai.get_status(), AIStatus::Ready);
}

#[test_case]
fn test_keyboard_input_processing() {
    let mut ai = AISystem::new();
    let _ = ai.initialize();
    
    ai.process_keyboard_input('a');
    ai.process_keyboard_input('b');
    
    // Should have added characters to input buffer
    assert_eq!(ai.input_buffer.len(), 2);
}