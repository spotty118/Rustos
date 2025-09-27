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
        // Initialize empty rules container - rules will be dynamically learned
        // from hardware patterns and system behavior during runtime
        self.rules.clear();
        crate::println!("[AI] Inference rules initialized - ready for dynamic rule learning");
        Ok(())
    }

    pub fn get_rules_count(&self) -> usize {
        self.rules.len()
    }

    pub fn learn_rule_from_pattern(&mut self, pattern: [f32; 8], confidence: f32, action: u8) -> Result<(), &'static str> {
        // Learn a new inference rule from observed patterns
        let rule = InferenceRule::new(pattern, confidence, action);
        self.add_rule(rule)?;
        Ok(())
    }

    pub fn adapt_rules_from_hardware(&mut self, metrics: &crate::ai::learning::HardwareMetrics) -> Result<(), &'static str> {
        // Dynamically create rules based on current hardware performance patterns
        let mut hardware_pattern = [0.0f32; 8];
        hardware_pattern[0] = (metrics.cpu_usage as f32) / 100.0;
        hardware_pattern[1] = (metrics.memory_usage as f32) / 100.0;
        hardware_pattern[2] = (metrics.io_operations as f32) / 1000.0;
        hardware_pattern[3] = (metrics.interrupt_count as f32) / 10000.0;
        hardware_pattern[4] = (metrics.context_switches as f32) / 1000.0;
        hardware_pattern[5] = (metrics.cache_misses as f32) / 10000.0;
        hardware_pattern[6] = (metrics.thermal_state as f32) / 100.0;
        hardware_pattern[7] = (metrics.power_efficiency as f32) / 100.0;

        // Determine confidence and action based on performance characteristics
        let performance_score = (100 - metrics.cpu_usage + 100 - metrics.memory_usage + metrics.power_efficiency) as f32 / 300.0;
        let confidence = if performance_score > 0.8 { 0.9 } else if performance_score > 0.6 { 0.7 } else { 0.5 };
        
        // Action codes: 1=optimize, 2=balance, 3=throttle
        let action = if performance_score > 0.8 { 1 } else if performance_score > 0.4 { 2 } else { 3 };
        
        self.learn_rule_from_pattern(hardware_pattern, confidence, action)?;
        Ok(())
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.max(0.0).min(1.0);
    }

    pub fn get_threshold(&self) -> f32 {
        self.threshold
    }
}

