use heapless::Vec;
use spin::Mutex;
use lazy_static::lazy_static;

pub mod neural_network;
pub mod inference_engine;
pub mod learning;
pub mod hardware_monitor;

use neural_network::NeuralNetwork;
use inference_engine::InferenceEngine;
use learning::{LearningSystem, HardwareMetrics};

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
    learning_system: LearningSystem,
    learned_patterns: Vec<[f32; MAX_INPUT_SIZE], MAX_LEARNED_PATTERNS>,
}

impl AISystem {
    pub fn new() -> Self {
        Self {
            status: AIStatus::Initializing,
            neural_network: NeuralNetwork::new(),
            inference_engine: InferenceEngine::new(),
            learning_system: LearningSystem::new(),
            learned_patterns: Vec::new(),
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

    pub fn process_hardware_data(&mut self, metrics: HardwareMetrics) {
        self.status = AIStatus::Learning;
        
        // Learn from hardware performance patterns
        if let Err(e) = self.learning_system.learn_from_hardware_metrics(&metrics) {
            crate::println!("[AI] Hardware learning error: {}", e);
            self.status = AIStatus::Error;
            return;
        }
        
        // Try to predict optimization strategy
        if let Some(optimization) = self.learning_system.predict_hardware_optimization(&metrics) {
            crate::println!("[AI] Predicted optimization: {:?}", optimization);
            hardware_monitor::apply_optimization(optimization);
        }
        
        self.status = AIStatus::Ready;
    }

    pub fn periodic_task(&mut self) {
        // Perform background AI tasks like hardware analysis
        if self.status == AIStatus::Ready {
            self.status = AIStatus::Inferencing;
            let _ = self.run_hardware_inference();
            self.status = AIStatus::Ready;
        }
    }



    fn run_hardware_inference(&mut self) -> Result<f32, &'static str> {
        let current_metrics = hardware_monitor::get_current_metrics();
        
        // Convert hardware metrics to neural network format
        let mut input = [0.0f32; MAX_INPUT_SIZE];
        input[0] = (current_metrics.cpu_usage as f32) / 100.0;
        input[1] = (current_metrics.memory_usage as f32) / 100.0;
        input[2] = (current_metrics.io_operations as f32) / 1000.0;
        input[3] = (current_metrics.interrupt_count as f32) / 10000.0;

        // Run inference
        let result = self.inference_engine.infer(&input)?;
        
        if result > 0.7 {
            crate::println!("[AI] High confidence hardware pattern detected: {:.2}", result);
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

pub fn process_hardware_metrics(metrics: HardwareMetrics) {
    let mut ai_system = AI_SYSTEM.lock();
    ai_system.process_hardware_data(metrics);
}

pub fn periodic_ai_task() {
    // Update hardware metrics and perform AI analysis
    let metrics = hardware_monitor::update_and_get_metrics();
    process_hardware_metrics(metrics);
    
    // Perform background AI tasks
    AI_SYSTEM.lock().periodic_task();
}

#[test_case]
fn test_ai_initialization() {
    let mut ai = AISystem::new();
    assert!(ai.initialize().is_ok());
    assert_eq!(ai.get_status(), AIStatus::Ready);
}

#[test_case]
fn test_hardware_metrics_processing() {
    let mut ai = AISystem::new();
    let _ = ai.initialize();
    
    let test_metrics = HardwareMetrics {
        cpu_usage: 50,
        memory_usage: 60,
        io_operations: 100,
        interrupt_count: 500,
        context_switches: 200,
        cache_misses: 50,
        thermal_state: 40,
        power_efficiency: 80,
    };
    
    ai.process_hardware_data(test_metrics);
    assert_eq!(ai.get_status(), AIStatus::Ready);
}