use super::neural_network::NeuralNetwork;
use heapless::Vec;
use micromath;

const MAX_RULES: usize = 16;

#[derive(Debug, Clone)]
pub struct InferenceRule {
    pub pattern: [f32; 8],
    pub confidence: f32,
    pub action: u8,
}

impl InferenceRule {
    pub fn new(pattern: [f32; 8], confidence: f32, action: u8) -> Self {
        Self {
            pattern,
            confidence,
            action,
        }
    }

    pub fn matches(&self, input: &[f32]) -> f32 {
        let mut similarity = 0.0;
        let mut count = 0;
        
        for (i, &val) in input.iter().enumerate().take(8) {
            let diff = (self.pattern[i] - val).abs();
            similarity += 1.0 - diff;
            count += 1;
        }
        
        if count > 0 {
            similarity / (count as f32)
        } else {
            0.0
        }
    }
}

pub struct InferenceEngine {
    neural_network: NeuralNetwork,
    rules: Vec<InferenceRule, MAX_RULES>,
    threshold: f32,
}

impl InferenceEngine {
    pub fn new() -> Self {
        Self {
            neural_network: NeuralNetwork::new(),
            rules: Vec::new(),
            threshold: 0.5,
        }
    }

    pub fn initialize(&mut self) -> Result<(), &'static str> {
        // Initialize the neural network
        self.neural_network.initialize()?;
        
        // Add some default inference rules
        self.add_default_rules()?;
        
        Ok(())
    }

    pub fn infer(&mut self, input: &[f32]) -> Result<f32, &'static str> {
        // First, try rule-based inference
        let rule_confidence = self.rule_based_inference(input)?;
        
        // Then, use neural network inference
        let nn_confidence = self.neural_network_inference(input)?;
        
        // Combine both approaches
        let combined_confidence = (rule_confidence + nn_confidence) / 2.0;
        
        Ok(combined_confidence)
    }

    fn rule_based_inference(&self, input: &[f32]) -> Result<f32, &'static str> {
        let mut max_confidence = 0.0;
        
        for rule in &self.rules {
            let similarity = rule.matches(input);
            let confidence = similarity * rule.confidence;
            
            if confidence > max_confidence {
                max_confidence = confidence;
            }
        }
        
        Ok(max_confidence)
    }

    fn neural_network_inference(&mut self, input: &[f32]) -> Result<f32, &'static str> {
        let prediction = self.neural_network.predict(input);
        
        // Apply sigmoid activation to get probability
        let confidence = 1.0 / (1.0 + micromath::F32Ext::exp(-prediction));
        
        Ok(confidence)
    }

    pub fn add_rule(&mut self, rule: InferenceRule) -> Result<(), &'static str> {
        self.rules.push(rule).map_err(|_| "Cannot add more rules")?;
        Ok(())
    }

    fn add_default_rules(&mut self) -> Result<(), &'static str> {
        // Add some example rules for pattern recognition
        
        // Rule 1: Detect ascending pattern
        let ascending_rule = InferenceRule::new(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            0.9,
            1, // Action code for ascending pattern
        );
        self.add_rule(ascending_rule)?;
        
        // Rule 2: Detect descending pattern
        let descending_rule = InferenceRule::new(
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            0.9,
            2, // Action code for descending pattern
        );
        self.add_rule(descending_rule)?;
        
        // Rule 3: Detect repetitive pattern
        let repetitive_rule = InferenceRule::new(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            0.8,
            3, // Action code for repetitive pattern
        );
        self.add_rule(repetitive_rule)?;
        
        // Rule 4: Detect high-activity pattern
        let high_activity_rule = InferenceRule::new(
            [0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8],
            0.7,
            4, // Action code for high activity
        );
        self.add_rule(high_activity_rule)?;
        
        Ok(())
    }

    pub fn get_rules_count(&self) -> usize {
        self.rules.len()
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.max(0.0).min(1.0);
    }

    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }
}

#[test_case]
fn test_inference_engine_creation() {
    let mut engine = InferenceEngine::new();
    assert!(engine.initialize().is_ok());
    assert!(engine.get_rules_count() > 0);
}

#[test_case]
fn test_inference_engine_rule_matching() {
    let mut engine = InferenceEngine::new();
    let _ = engine.initialize();
    
    // Test ascending pattern
    let ascending_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let confidence = engine.infer(&ascending_input).unwrap();
    
    assert!(confidence > 0.0);
}

#[test_case]
fn test_inference_rule_similarity() {
    let rule = InferenceRule::new([0.5; 8], 1.0, 1);
    let input = [0.5; 8];
    let similarity = rule.matches(&input);
    
    assert!((similarity - 1.0).abs() < 0.001); // Should be very close to 1.0
}