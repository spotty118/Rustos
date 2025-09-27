use heapless::Vec;
use micromath;

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
        
        // Calculate expected optimization score based on current performance and historical data
        let expected_output = self.calculate_adaptive_performance_score(metrics);
        
        let sample = TrainingSample::new(input_pattern, expected_output);
        self.add_training_sample(sample)?;
        
        // Update learning parameters based on system performance
        self.adapt_to_system_state(metrics);
        
        Ok(())
    }
    
    /// Calculate adaptive performance score using historical trends
    fn calculate_adaptive_performance_score(&self, metrics: &HardwareMetrics) -> f32 {
        // Base performance score
        let cpu_score = 1.0 - (metrics.cpu_usage as f32 / 100.0);
        let memory_score = 1.0 - (metrics.memory_usage as f32 / 100.0);
        let thermal_score = 1.0 - (metrics.thermal_state as f32 / 100.0);
        let power_score = metrics.power_efficiency as f32 / 100.0;
        
        let base_score = (cpu_score + memory_score + thermal_score + power_score) / 4.0;
        
        // Apply adaptive weighting based on historical patterns
        let adaptive_weight = self.calculate_adaptive_weight(metrics);
        
        // Consider system stability over time
        let stability_factor = self.calculate_stability_factor();
        
        (base_score * adaptive_weight + stability_factor) / 2.0
    }
    
    /// Calculate adaptive weight based on system patterns
    fn calculate_adaptive_weight(&self, metrics: &HardwareMetrics) -> f32 {
        let mut weight: f32 = 1.0;
        
        // Increase weight for critical resource usage patterns
        if metrics.cpu_usage > 80 {
            weight *= 1.2; // High CPU usage needs attention
        }
        
        if metrics.memory_usage > 90 {
            weight *= 1.3; // Critical memory usage
        }
        
        if metrics.thermal_state > 85 {
            weight *= 1.4; // Thermal throttling risk
        }
        
        // Reduce weight for already efficient systems
        if metrics.power_efficiency > 80 {
            weight *= 0.9;
        }
        
        weight.min(2.0).max(0.5) // Keep weight in reasonable bounds
    }
    
    /// Calculate system stability factor from training history
    fn calculate_stability_factor(&self) -> f32 {
        if self.training_samples.len() < 3 {
            return 0.5; // Default stability for new systems
        }
        
        // Calculate variance in recent performance scores
        let recent_samples = self.training_samples.len().min(5);
        let start_idx = self.training_samples.len() - recent_samples;
        
        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        
        for i in start_idx..self.training_samples.len() {
            let score = self.training_samples[i].expected_output;
            sum += score;
            sum_squares += score * score;
        }
        
        let mean = sum / recent_samples as f32;
        let variance = (sum_squares / recent_samples as f32) - (mean * mean);
        
        // Lower variance indicates more stability
        1.0 - variance.min(1.0)
    }
    
    /// Adapt learning parameters to current system state
    fn adapt_to_system_state(&mut self, metrics: &HardwareMetrics) {
        // Increase learning rate for systems under stress
        if metrics.cpu_usage > 90 || metrics.memory_usage > 90 || metrics.thermal_state > 90 {
            self.learning_rate *= 1.1;
        }
        
        // Decrease learning rate for stable systems
        if metrics.cpu_usage < 30 && metrics.memory_usage < 50 && metrics.thermal_state < 40 {
            self.learning_rate *= 0.95;
        }
        
        // Keep learning rate within bounds
        self.learning_rate = self.learning_rate.max(0.001).min(0.1);
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
            // Use multiple similarity metrics for better pattern detection
            let euclidean_similarity = self.calculate_euclidean_similarity(input, &sample.input);
            let cosine_similarity = self.calculate_cosine_similarity(input, &sample.input);
            let manhattan_similarity = self.calculate_manhattan_similarity(input, &sample.input);
            
            // Weighted combination of similarity metrics
            let combined_similarity = euclidean_similarity * 0.4 + 
                                     cosine_similarity * 0.4 + 
                                     manhattan_similarity * 0.2;
            
            // Apply temporal weighting (more recent samples get higher weight)
            let temporal_weight = 1.0 - (i as f32 / self.training_samples.len() as f32) * 0.2;
            let final_similarity = combined_similarity * temporal_weight;
            
            if final_similarity > 0.6 { // Lowered threshold for better sensitivity
                let _ = matches.push((i, final_similarity));
                if matches.len() >= 8 {
                    break;
                }
            }
        }

        // Sort by similarity (highest first) using stable sort
        self.stable_sort_by_similarity(&mut matches);
        
        matches
    }
    
    /// Calculate Euclidean distance-based similarity
    fn calculate_euclidean_similarity(&self, pattern1: &[f32], pattern2: &[f32]) -> f32 {
        if pattern1.len() != pattern2.len() {
            return 0.0;
        }

        let mut sum_squares = 0.0;
        for (a, b) in pattern1.iter().zip(pattern2.iter()) {
            let diff = a - b;
            sum_squares += diff * diff;
        }
        
        let distance = micromath::F32Ext::sqrt(sum_squares);
        // Convert distance to similarity (0-1 range)
        1.0 / (1.0 + distance)
    }
    
    /// Calculate cosine similarity
    fn calculate_cosine_similarity(&self, pattern1: &[f32], pattern2: &[f32]) -> f32 {
        if pattern1.len() != pattern2.len() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        
        for (a, b) in pattern1.iter().zip(pattern2.iter()) {
            dot_product += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        
        let cosine = dot_product / (micromath::F32Ext::sqrt(norm1) * micromath::F32Ext::sqrt(norm2));
        (cosine + 1.0) / 2.0 // Normalize to 0-1 range
    }
    
    /// Calculate Manhattan distance-based similarity
    fn calculate_manhattan_similarity(&self, pattern1: &[f32], pattern2: &[f32]) -> f32 {
        if pattern1.len() != pattern2.len() {
            return 0.0;
        }

        let mut sum_abs_diff = 0.0;
        for (a, b) in pattern1.iter().zip(pattern2.iter()) {
            sum_abs_diff += (a - b).abs();
        }
        
        // Convert distance to similarity
        1.0 / (1.0 + sum_abs_diff)
    }
    
    /// Stable sort matches by similarity (highest first)
    fn stable_sort_by_similarity(&self, matches: &mut Vec<(usize, f32), 8>) {
        // Insertion sort for stability with small arrays
        for i in 1..matches.len() {
            let key = matches[i];
            let mut j = i;
            
            while j > 0 && matches[j - 1].1 < key.1 {
                matches[j] = matches[j - 1];
                j -= 1;
            }
            
            matches[j] = key;
        }
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