use heapless::Vec;

const MAX_TRAINING_SAMPLES: usize = 32;
const MAX_PATTERN_SIZE: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct HardwareMetrics {
    pub cpu_usage: u8,           // CPU usage percentage (0-100)
    pub memory_usage: u8,        // Memory usage percentage (0-100)
    pub io_operations: u32,      // Number of I/O operations per second
    pub interrupt_count: u32,    // Interrupts per second
    pub context_switches: u32,   // Context switches per second
    pub cache_misses: u32,       // Cache misses per second
    pub thermal_state: u8,       // Thermal state (0-100, higher = hotter)
    pub power_efficiency: u8,    // Power efficiency score (0-100, higher = better)
    pub gpu_usage: u8,           // GPU usage percentage (0-100) - NEW
    pub gpu_memory_usage: u8,    // GPU memory usage percentage (0-100) - NEW
    pub gpu_temperature: u8,     // GPU temperature relative (0-100) - NEW
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HardwareOptimization {
    OptimalPerformance,    // High performance, higher power usage
    BalancedMode,         // Balanced performance and power
    PowerSaving,          // Lower performance, power efficient
    ThermalThrottle,      // Reduce performance to manage heat
}

#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub input: [f32; MAX_PATTERN_SIZE],
    pub expected_output: f32,
    pub weight: f32,
}

impl TrainingSample {
    pub fn new(input: [f32; MAX_PATTERN_SIZE], expected_output: f32) -> Self {
        Self {
            input,
            expected_output,
            weight: 1.0,
        }
    }

    pub fn with_weight(input: [f32; MAX_PATTERN_SIZE], expected_output: f32, weight: f32) -> Self {
        Self {
            input,
            expected_output,
            weight,
        }
    }
}

pub struct LearningSystem {
    training_samples: Vec<TrainingSample, MAX_TRAINING_SAMPLES>,
    learning_rate: f32,
    momentum: f32,
    convergence_threshold: f32,
}

impl LearningSystem {
    pub fn new() -> Self {
        Self {
            training_samples: Vec::new(),
            learning_rate: 0.01,
            momentum: 0.9,
            convergence_threshold: 0.001,
        }
    }

    pub fn add_training_sample(&mut self, sample: TrainingSample) -> Result<(), &'static str> {
        if self.training_samples.len() >= MAX_TRAINING_SAMPLES {
            // Remove oldest sample to make room for new one
            self.training_samples.remove(0);
        }
        
        self.training_samples.push(sample).map_err(|_| "Failed to add training sample")?;
        Ok(())
    }

    pub fn learn_from_hardware_metrics(&mut self, metrics: &HardwareMetrics) -> Result<(), &'static str> {
        // Create training pattern from hardware metrics
        let mut input_pattern = [0.0f32; MAX_PATTERN_SIZE];
        
        // Normalize hardware metrics to 0-1 range for neural network input
        input_pattern[0] = (metrics.cpu_usage as f32) / 100.0;           // CPU usage percentage
        input_pattern[1] = (metrics.memory_usage as f32) / 100.0;        // Memory usage percentage
        input_pattern[2] = (metrics.io_operations as f32) / 1000.0;      // Normalized I/O ops
        input_pattern[3] = (metrics.interrupt_count as f32) / 10000.0;   // Normalized interrupt count
        input_pattern[4] = (metrics.context_switches as f32) / 1000.0;   // Normalized context switches
        input_pattern[5] = (metrics.cache_misses as f32) / 10000.0;      // Normalized cache misses
        input_pattern[6] = (metrics.thermal_state as f32) / 100.0;       // Thermal state (0-100)
        input_pattern[7] = (metrics.power_efficiency as f32) / 100.0;    // Power efficiency (0-100)
        
        // Calculate expected optimization score based on current performance
        let expected_output = self.calculate_performance_score(metrics);
        
        let sample = TrainingSample::new(input_pattern, expected_output);
        self.add_training_sample(sample)?;
        
        Ok(())
    }
    
    fn calculate_performance_score(&self, metrics: &HardwareMetrics) -> f32 {
        // Higher score for better performance (lower usage, higher efficiency)
        let cpu_score = 1.0 - (metrics.cpu_usage as f32 / 100.0);
        let memory_score = 1.0 - (metrics.memory_usage as f32 / 100.0);
        let thermal_score = 1.0 - (metrics.thermal_state as f32 / 100.0);
        let power_score = metrics.power_efficiency as f32 / 100.0;
        
        (cpu_score + memory_score + thermal_score + power_score) / 4.0
    }

    pub fn learn_pattern_recognition(&mut self, patterns: &[[f32; MAX_PATTERN_SIZE]], labels: &[f32]) -> Result<(), &'static str> {
        if patterns.len() != labels.len() {
            return Err("Patterns and labels must have same length");
        }

        for (pattern, &label) in patterns.iter().zip(labels.iter()) {
            let sample = TrainingSample::new(*pattern, label);
            self.add_training_sample(sample)?;
        }

        Ok(())
    }

    pub fn calculate_pattern_similarity(&self, pattern1: &[f32], pattern2: &[f32]) -> f32 {
        if pattern1.len() != pattern2.len() {
            return 0.0;
        }

        let mut similarity = 0.0;
        for (a, b) in pattern1.iter().zip(pattern2.iter()) {
            similarity += 1.0 - (a - b).abs();
        }

        similarity / (pattern1.len() as f32)
    }

    pub fn detect_patterns(&self, input: &[f32; MAX_PATTERN_SIZE]) -> Vec<(usize, f32), 8> {
        let mut matches = Vec::new();

        for (i, sample) in self.training_samples.iter().enumerate() {
            let similarity = self.calculate_pattern_similarity(input, &sample.input);
            
            if similarity > 0.7 { // High similarity threshold
                let _ = matches.push((i, similarity));
                if matches.len() >= 8 {
                    break;
                }
            }
        }

        // Sort by similarity (highest first) - simple bubble sort
        for i in 0..matches.len() {
            for j in 0..matches.len() - 1 - i {
                if matches[j].1 < matches[j + 1].1 {
                    let temp = matches[j];
                    matches[j] = matches[j + 1];
                    matches[j + 1] = temp;
                }
            }
        }
        
        matches
    }

    pub fn predict_next_pattern(&self, current_input: &[f32; MAX_PATTERN_SIZE]) -> Option<f32> {
        let matches = self.detect_patterns(current_input);
        
        if matches.is_empty() {
            return None;
        }
        
        // Use the most similar pattern's expected output
        let best_match_idx = matches[0].0;
        if let Some(sample) = self.training_samples.get(best_match_idx) {
            Some(sample.expected_output)
        } else {
            None
        }
    }

    pub fn predict_hardware_optimization(&self, current_metrics: &HardwareMetrics) -> Option<HardwareOptimization> {
        let mut input_pattern = [0.0f32; MAX_PATTERN_SIZE];
        
        // Convert current metrics to input pattern
        input_pattern[0] = (current_metrics.cpu_usage as f32) / 100.0;
        input_pattern[1] = (current_metrics.memory_usage as f32) / 100.0;
        input_pattern[2] = (current_metrics.io_operations as f32) / 1000.0;
        input_pattern[3] = (current_metrics.interrupt_count as f32) / 10000.0;
        input_pattern[4] = (current_metrics.context_switches as f32) / 1000.0;
        input_pattern[5] = (current_metrics.cache_misses as f32) / 10000.0;
        input_pattern[6] = (current_metrics.thermal_state as f32) / 100.0;
        input_pattern[7] = (current_metrics.power_efficiency as f32) / 100.0;
        
        // Find best matching pattern
        if let Some(prediction) = self.predict_next_pattern(&input_pattern) {
            // Convert prediction to optimization strategy
            if prediction > 0.8 {
                Some(HardwareOptimization::OptimalPerformance)
            } else if prediction > 0.6 {
                Some(HardwareOptimization::BalancedMode)
            } else if prediction > 0.4 {
                Some(HardwareOptimization::PowerSaving)
            } else {
                Some(HardwareOptimization::ThermalThrottle)
            }
        } else {
            None
        }
    }

    pub fn adapt_learning_rate(&mut self, error: f32) {
        // Adaptive learning rate based on error
        if error > 0.1 {
            self.learning_rate *= 1.1; // Increase learning rate for high error
        } else if error < 0.01 {
            self.learning_rate *= 0.95; // Decrease learning rate for low error
        }

        // Keep learning rate within reasonable bounds
        self.learning_rate = self.learning_rate.max(0.001).min(0.1);
    }

    pub fn get_training_samples_count(&self) -> usize {
        self.training_samples.len()
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate.max(0.0).min(1.0);
    }

    pub fn clear_training_data(&mut self) {
        self.training_samples.clear();
    }
}

#[test_case]
fn test_learning_system_creation() {
    let learning_system = LearningSystem::new();
    assert_eq!(learning_system.get_training_samples_count(), 0);
    assert!(learning_system.get_learning_rate() > 0.0);
}

#[test_case]
fn test_add_training_sample() {
    let mut learning_system = LearningSystem::new();
    let sample = TrainingSample::new([0.1; MAX_PATTERN_SIZE], 0.5);
    
    assert!(learning_system.add_training_sample(sample).is_ok());
    assert_eq!(learning_system.get_training_samples_count(), 1);
}

#[test_case]
fn test_pattern_similarity() {
    let learning_system = LearningSystem::new();
    let pattern1 = [0.5; MAX_PATTERN_SIZE];
    let pattern2 = [0.5; MAX_PATTERN_SIZE];
    
    let similarity = learning_system.calculate_pattern_similarity(&pattern1, &pattern2);
    assert!((similarity - 1.0).abs() < 0.001);
}