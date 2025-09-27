use heapless::Vec;

const MAX_TRAINING_SAMPLES: usize = 32;
const MAX_PATTERN_SIZE: usize = 8;

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

    pub fn learn_from_keyboard_input(&mut self, input_sequence: &[u8]) -> Result<(), &'static str> {
        if input_sequence.len() < 2 {
            return Ok(()); // Need at least 2 characters for learning
        }

        // Convert input sequence to training samples
        for window in input_sequence.windows(MAX_PATTERN_SIZE + 1) {
            if window.len() == MAX_PATTERN_SIZE + 1 {
                let mut input_pattern = [0.0f32; MAX_PATTERN_SIZE];
                
                // Normalize input characters to 0-1 range
                for (i, &byte) in window[..MAX_PATTERN_SIZE].iter().enumerate() {
                    input_pattern[i] = (byte as f32) / 255.0;
                }
                
                // Use next character as expected output
                let expected_output = (window[MAX_PATTERN_SIZE] as f32) / 255.0;
                
                let sample = TrainingSample::new(input_pattern, expected_output);
                self.add_training_sample(sample)?;
            }
        }

        Ok(())
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

        // Weighted average of similar patterns
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (sample_idx, similarity) in matches {
            if let Some(sample) = self.training_samples.get(sample_idx) {
                weighted_sum += sample.expected_output * similarity * sample.weight;
                total_weight += similarity * sample.weight;
            }
        }

        if total_weight > 0.0 {
            Some(weighted_sum / total_weight)
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