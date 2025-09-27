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
        crate::println!("[AI] Initializing production neural network...");
        self.neural_network.initialize()?;
        
        crate::println!("[AI] Initializing production inference engine...");
        self.inference_engine.initialize()?;
        
        crate::println!("[AI] Preparing dynamic pattern learning system...");
        self.load_default_patterns()?;
        
        self.status = AIStatus::Ready;
        crate::println!("[AI] Production AI system successfully initialized!");
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

        // Dynamically adapt inference rules based on current hardware patterns
        if let Err(e) = self.inference_engine.adapt_rules_from_hardware(&metrics) {
            crate::println!("[AI] Rule adaptation error: {}", e);
        }

        // Store learned patterns from hardware metrics for future reference
        self.store_hardware_pattern(&metrics);
        
        // Try to predict optimization strategy
        if let Some(optimization) = self.learning_system.predict_hardware_optimization(&metrics) {
            crate::println!("[AI] Predicted optimization: {:?}", optimization);
            hardware_monitor::apply_optimization(optimization);
        }
        
        self.status = AIStatus::Ready;
    }

    fn store_hardware_pattern(&mut self, metrics: &HardwareMetrics) {
        // Convert hardware metrics to a learnable pattern
        let mut pattern = [0.0f32; MAX_INPUT_SIZE];
        pattern[0] = (metrics.cpu_usage as f32) / 100.0;
        pattern[1] = (metrics.memory_usage as f32) / 100.0;
        pattern[2] = (metrics.io_operations as f32) / 1000.0;
        pattern[3] = (metrics.interrupt_count as f32) / 10000.0;
        pattern[4] = (metrics.context_switches as f32) / 1000.0;
        pattern[5] = (metrics.cache_misses as f32) / 10000.0;
        pattern[6] = (metrics.thermal_state as f32) / 100.0;
        pattern[7] = (metrics.power_efficiency as f32) / 100.0;

        // Store pattern if there's space
        if self.learned_patterns.len() < MAX_LEARNED_PATTERNS {
            let _ = self.learned_patterns.push(pattern);
        } else {
            // Replace oldest pattern with new one
            self.learned_patterns.remove(0);
            let _ = self.learned_patterns.push(pattern);
        }
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
        // Initialize empty patterns container - patterns will be learned during runtime
        // from actual hardware metrics and system behavior
        self.learned_patterns.clear();
        crate::println!("[AI] Pattern storage initialized - ready to learn from hardware");
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

