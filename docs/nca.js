/**
 * Neural Cellular Automata (NCA) model implementation
 */

import { operators } from './sph.js';
import { mathUtils } from './utils.js';

export class NCA {
    constructor(weights) {
        this.weights = weights;
        this.config = weights.config;
        
        // Validate weights structure
        if (!weights.layers || weights.layers.length < 2) {
            throw new Error('Invalid weights: expected at least 2 layers');
        }
        
        // Extract layer weights and biases
        this.layer1_weight = weights.layers[0].weight;
        this.layer1_bias = weights.layers[0].bias;
        this.layer2_weight = weights.layers[1].weight;
        this.layer2_bias = weights.layers[1].bias;
        
        // Model configuration
        this.inputFeatures = this.config.input_features;
        this.hiddenFeatures = this.config.hidden_features;
        this.outputFeatures = this.config.output_features;
        this.fireRate = this.config.fire_rate;
        this.updateRule = this.config.update_rule;
        this.h = this.config.h;
        
        // Infer number of cell features from input/output dimensions
        // Input is [A, gA_x, gA_y] so cell features = input_features / 3
        this.cellFeatures = Math.floor(this.inputFeatures / 3);
        
        console.log('NCA Model initialized:');
        console.log(`  Cell features: ${this.cellFeatures}`);
        console.log(`  Input features: ${this.inputFeatures}`);
        console.log(`  Hidden features: ${this.hiddenFeatures}`);
        console.log(`  Output features: ${this.outputFeatures}`);
        console.log(`  Fire rate: ${this.fireRate}`);
        console.log(`  Update rule: ${this.updateRule}`);
    }
    
    /**
     * Linear layer forward pass
     * @param {Array} input - Input vector
     * @param {Array} weight - Weight matrix [output_size x input_size]
     * @param {Array} bias - Bias vector [output_size]
     * @returns {Array} Output vector
     */
    _linear(input, weight, bias) {
        const output = new Array(weight.length);
        
        for (let i = 0; i < weight.length; i++) {
            let sum = bias[i];
            for (let j = 0; j < input.length; j++) {
                sum += weight[i][j] * input[j];
            }
            output[i] = sum;
        }
        
        return output;
    }
    
    /**
     * ReLU activation function
     * @param {Array} x - Input vector
     * @returns {Array} Output vector
     */
    _relu(x) {
        return x.map(val => Math.max(0, val));
    }
    
    /**
     * Sigmoid activation function
     * @param {Array} x - Input vector
     * @returns {Array} Output vector
     */
    _sigmoid(x) {
        return x.map(val => 1 / (1 + Math.exp(-val)));
    }
    
    /**
     * Tanh activation function
     * @param {Array} x - Input vector
     * @returns {Array} Output vector
     */
    _tanh(x) {
        return x.map(val => Math.tanh(val));
    }
    
    /**
     * Perceive function - calculates gradients using SPH
     * @param {Array} points - Particle positions
     * @param {Array} volumes - Particle volumes
     * @param {Array} attributes - Particle attributes
     * @param {number} h - Kernel radius
     * @param {Object} hashGrid - Hash grid for neighbor finding
     * @returns {Array} Perceived gradients [N x F x 2]
     */
    _perceive(points, volumes, attributes, h, hashGrid) {
        return operators.calculate_gradient(points, attributes, volumes, hashGrid, h);
    }
    
    /**
     * Calculate life mask - determines which particles are alive
     * @param {Array} points - Particle positions
     * @param {Array} volumes - Particle volumes
     * @param {Array} attributes - Particle attributes
     * @param {number} h - Kernel radius
     * @param {Object} hashGrid - Hash grid for neighbor finding
     * @returns {Array} Life mask [N]
     */
    _lifeMask(points, volumes, attributes, h, hashGrid) {
        const numPoints = points.length;
        const mask = new Array(numPoints);
        
        // Initial mask based on activity threshold
        for (let i = 0; i < numPoints; i++) {
            mask[i] = attributes[i][3] > 0.1? 1.0: 0.0; // Alpha > 0.1
        }
        
        // Apply smoothing using SPH blur
        const activityArray = mask.map(a => [a]);
        const smoothedActivity = operators.apply_blur(points, activityArray, volumes, hashGrid, h);
        
        // Final mask based on smoothed activity
        for (let i = 0; i < numPoints; i++) {
            mask[i] = smoothedActivity[i][0] > 0.1;
        }
        
        return mask;
    }
    
    /**
     * Forward pass of the NCA model
     * @param {Array} points - Particle positions
     * @param {Array} volumes - Particle volumes
     * @param {Array} attributes - Particle attributes [N x F]
     * @param {number} h - Kernel radius
     * @param {Object} hashGrid - Hash grid for neighbor finding
     * @param {number} fireRate - Override fire rate (optional)
     * @returns {Array} Updated attributes [N x F]
     */
    forward(points, volumes, attributes, h, hashGrid, fireRate = null, useAlpha = true) {
        const numPoints = points.length;
        const currentFireRate = fireRate !== null ? fireRate : this.fireRate;
        
        // Calculate life mask before update
        let prevMask = null;
        if (useAlpha) {
            prevMask = this._lifeMask(points, volumes, attributes, h, hashGrid);
        }
        
        // Perceive - calculate gradients
        const gradients = this._perceive(points, volumes, attributes, h, hashGrid);
        
        // Prepare input for neural network
        const newAttributes = new Array(numPoints);
        
        for (let i = 0; i < numPoints; i++) {
            // Concatenate [A, gA_x, gA_y] for input
            const input = [];
            
            // Add attributes
            for (let f = 0; f < this.cellFeatures; f++) {
                input.push(attributes[i][f] || 0);
            }
            
            // Add gradient x components
            for (let f = 0; f < this.cellFeatures; f++) {
                input.push(gradients[i][f] ? gradients[i][f][0] * h / this.h: 0);
            }
            
            // Add gradient y components
            for (let f = 0; f < this.cellFeatures; f++) {
                input.push(gradients[i][f] ? gradients[i][f][1] * h / this.h : 0);
            }
            
            // Neural network forward pass
            const hidden = this._relu(this._linear(input, this.layer1_weight, this.layer1_bias));
            const output = this._linear(hidden, this.layer2_weight, this.layer2_bias);
            
            // Apply update rule
            let newA;
            if (this.updateRule === 'gated') {
                // Gated update: A_new = A * gate + delta * mult
                const gate = this._sigmoid(output.slice(0, this.cellFeatures));
                const delta = this._tanh(output.slice(this.cellFeatures, 2 * this.cellFeatures));
                const mult = this._sigmoid(output.slice(-1)); // Last element
                
                newA = new Array(this.cellFeatures);
                for (let f = 0; f < this.cellFeatures; f++) {
                    newA[f] = attributes[i][f] * gate[f] + delta[f] * mult[0];
                }
            } else {
                // Original update rule: A_new = A + dA * fire_rate
                newA = new Array(this.cellFeatures);
                for (let f = 0; f < this.cellFeatures; f++) {
                    newA[f] = attributes[i][f] + output[f] * currentFireRate;
                }
            }
            
            // Apply fire rate mask
            const shouldUpdate = Math.random() <= currentFireRate;
            if (shouldUpdate) {
                newAttributes[i] = newA;
            } else {
                newAttributes[i] = [...attributes[i]];
            }
        }
        
        // Calculate new life mask
        let newMask = null;
        if (useAlpha) {
            newMask = this._lifeMask(points, volumes, newAttributes, h, hashGrid);
        
            // Apply living mask - only living cells survive
            for (let i = 0; i < numPoints; i++) {
                const isLiving = prevMask[i] && newMask[i];
                if (!isLiving) {
                    // Kill the cell
                    for (let f = 0; f < this.cellFeatures; f++) {
                        newAttributes[i][f] = 0;
                    }
                }
            }
        }
        
        return newAttributes;
    }
    
    /**
     * Convert attributes to RGBA for visualization
     * @param {Array} attributes - Particle attributes
     * @returns {Array} RGBA values [N x 4]
     */
    toRGBA(attributes) {
        const rgba = new Array(attributes.length);
        
        for (let i = 0; i < attributes.length; i++) {
            const attr = attributes[i];
            
            // Extract RGBA channels, clamping to [0, 1]
            const r = mathUtils.clamp(attr[0] || 0, 0, 1);
            const g = mathUtils.clamp(attr[1] || 0, 0, 1);
            const b = mathUtils.clamp(attr[2] || 0, 0, 1);
            const a = mathUtils.clamp(attr[3] || 0, 0, 1);
            
            rgba[i] = [r, g, b, a];
        }
        
        return rgba;
    }
    
    /**
     * Get model statistics
     */
    getStats() {
        return {
            cellFeatures: this.cellFeatures,
            inputFeatures: this.inputFeatures,
            hiddenFeatures: this.hiddenFeatures,
            outputFeatures: this.outputFeatures,
            fireRate: this.fireRate,
            updateRule: this.updateRule,
            totalParameters: this.layer1_weight.length * this.layer1_weight[0].length + 
                           this.layer1_bias.length +
                           this.layer2_weight.length * this.layer2_weight[0].length + 
                           this.layer2_bias.length
        };
    }
}
