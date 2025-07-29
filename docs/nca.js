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
        
        // Pre-allocate memory pools for batch processing
        this.maxParticles = 8192 * 4; // Support up to 8K particles
        this.inputBuffer = new Float32Array(this.maxParticles * this.inputFeatures);
        this.hiddenBuffer = new Float32Array(this.maxParticles * this.hiddenFeatures);
        this.outputBuffer = new Float32Array(this.maxParticles * this.outputFeatures);
        
        // Initialize TensorFlow.js tensors for GPU acceleration
        this.initializeTensorFlowTensors();
        
        console.log('NCA Model initialized:');
        console.log(`  Cell features: ${this.cellFeatures}`);
        console.log(`  Input features: ${this.inputFeatures}`);
        console.log(`  Hidden features: ${this.hiddenFeatures}`);
        console.log(`  Output features: ${this.outputFeatures}`);
        console.log(`  Fire rate: ${this.fireRate}`);
        console.log(`  Update rule: ${this.updateRule}`);
        console.log(`  Memory pools allocated for ${this.maxParticles} particles`);
        console.log(`  TensorFlow.js backend: ${typeof tf !== 'undefined' ? tf.getBackend() : 'not available'}`);
    }
    
    /**
     * Initialize TensorFlow.js tensors from weights
     */
    initializeTensorFlowTensors() {
        // Check if TensorFlow.js is available
        if (typeof tf === 'undefined') {
            console.warn('TensorFlow.js not available, using fallback implementation');
            this.useTensorFlow = false;
            return;
        }
        
        try {
            // Convert weight matrices to tensors (transposed for efficient matmul)
            this.layer1WeightTensor = tf.tensor2d(this.layer1_weight).transpose();
            this.layer1BiasTensor = tf.tensor1d(this.layer1_bias);
            this.layer2WeightTensor = tf.tensor2d(this.layer2_weight).transpose();
            this.layer2BiasTensor = tf.tensor1d(this.layer2_bias);
            
            this.useTensorFlow = true;
            console.log('TensorFlow.js tensors initialized successfully');
        } catch (error) {
            console.warn('Failed to initialize TensorFlow.js tensors:', error);
            this.useTensorFlow = false;
        }
    }
    
    /**
     * Dispose of TensorFlow.js tensors to free memory
     */
    dispose() {
        if (this.useTensorFlow) {
            this.layer1WeightTensor?.dispose();
            this.layer1BiasTensor?.dispose();
            this.layer2WeightTensor?.dispose();
            this.layer2BiasTensor?.dispose();
        }
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
     * Batch linear layer forward pass
     * @param {Float32Array} inputMatrix - Input matrix [numParticles * inputSize]
     * @param {Array} weight - Weight matrix [outputSize x inputSize]
     * @param {Array} bias - Bias vector [outputSize]
     * @param {number} numParticles - Number of particles
     * @param {Float32Array} outputMatrix - Pre-allocated output buffer
     * @returns {Float32Array} Output matrix [numParticles * outputSize]
     */
    _batchLinear(inputMatrix, weight, bias, numParticles, outputMatrix) {
        const inputSize = weight[0].length;
        const outputSize = weight.length;
        
        // Optimized matrix multiplication: Output = Input * Weight^T + Bias
        for (let p = 0; p < numParticles; p++) {
            for (let o = 0; o < outputSize; o++) {
                let sum = bias[o];
                for (let i = 0; i < inputSize; i++) {
                    sum += inputMatrix[p * inputSize + i] * weight[o][i];
                }
                outputMatrix[p * outputSize + o] = sum;
            }
        }
        
        return outputMatrix;
    }
    
    /**
     * Batch ReLU activation function
     * @param {Float32Array} matrix - Input/output matrix
     * @param {number} length - Total number of elements
     */
    _batchRelu(matrix, length) {
        for (let i = 0; i < length; i++) {
            matrix[i] = Math.max(0, matrix[i]);
        }
    }
    
    /**
     * Batch sigmoid activation function
     * @param {Float32Array} matrix - Input/output matrix
     * @param {number} length - Total number of elements
     */
    _batchSigmoid(matrix, length) {
        for (let i = 0; i < length; i++) {
            matrix[i] = 1 / (1 + Math.exp(-matrix[i]));
        }
    }
    
    /**
     * Batch tanh activation function
     * @param {Float32Array} matrix - Input/output matrix
     * @param {number} length - Total number of elements
     */
    _batchTanh(matrix, length) {
        for (let i = 0; i < length; i++) {
            matrix[i] = Math.tanh(matrix[i]);
        }
    }
    
    /**
     * Prepare input batch for neural network processing
     * @param {Array} attributes - Particle attributes [N x F]
     * @param {Array} gradients - Particle gradients [N x F x 2]
     * @param {number} numPoints - Number of particles
     * @param {number} h - Current kernel radius
     * @returns {Float32Array} Input matrix [numParticles * inputFeatures]
     */
    _prepareInputBatch(attributes, gradients, numPoints, h) {
        const inputMatrix = this.inputBuffer.subarray(0, numPoints * this.inputFeatures);
        
        // Vectorized copy of attributes, gradient_x, gradient_y
        for (let i = 0; i < numPoints; i++) {
            const offset = i * this.inputFeatures;
            
            // Copy attributes [A]
            for (let f = 0; f < this.cellFeatures; f++) {
                inputMatrix[offset + f] = attributes[i][f] || 0;
            }
            
            // Copy gradient_x [gA_x]
            for (let f = 0; f < this.cellFeatures; f++) {
                inputMatrix[offset + this.cellFeatures + f] = 
                    gradients[i][f] ? gradients[i][f][0] * h / this.h : 0;
            }
            
            // Copy gradient_y [gA_y]
            for (let f = 0; f < this.cellFeatures; f++) {
                inputMatrix[offset + 2 * this.cellFeatures + f] = 
                    gradients[i][f] ? gradients[i][f][1] * h / this.h : 0;
            }
        }
        
        return inputMatrix;
    }
    
    /**
     * Batch neural network forward pass (TensorFlow.js optimized)
     * @param {Float32Array} inputMatrix - Input matrix [numParticles * inputFeatures]
     * @param {number} numPoints - Number of particles
     * @returns {Float32Array} Output matrix [numParticles * outputFeatures]
     */
    _forwardBatch(inputMatrix, numPoints) {
        // Use TensorFlow.js if available, otherwise fallback to manual implementation
        if (this.useTensorFlow) {
            return tf.tidy(() => {
                // Create input tensor from Float32Array
                const inputTensor = tf.tensor2d(Array.from(inputMatrix.subarray(0, numPoints * this.inputFeatures)), [numPoints, this.inputFeatures]);
                
                // Layer 1: Input -> Hidden with ReLU
                const hidden = tf.relu(
                    tf.add(
                        tf.matMul(inputTensor, this.layer1WeightTensor),
                        this.layer1BiasTensor
                    )
                );
                
                // Layer 2: Hidden -> Output
                const output = tf.add(
                    tf.matMul(hidden, this.layer2WeightTensor),
                    this.layer2BiasTensor
                );
                
                // Convert back to Float32Array
                const outputData = output.dataSync();
                const outputMatrix = this.outputBuffer.subarray(0, numPoints * this.outputFeatures);
                outputMatrix.set(outputData);
                return outputMatrix;
            });
        } else {
            // Fallback to manual batch processing
            // Layer 1: Input -> Hidden
            const hiddenMatrix = this.hiddenBuffer.subarray(0, numPoints * this.hiddenFeatures);
            this._batchLinear(
                inputMatrix, 
                this.layer1_weight, 
                this.layer1_bias, 
                numPoints,
                hiddenMatrix
            );
            this._batchRelu(hiddenMatrix, numPoints * this.hiddenFeatures);
            
            // Layer 2: Hidden -> Output  
            const outputMatrix = this.outputBuffer.subarray(0, numPoints * this.outputFeatures);
            this._batchLinear(
                hiddenMatrix,
                this.layer2_weight,
                this.layer2_bias,
                numPoints,
                outputMatrix
            );
            
            return outputMatrix;
        }
    }
    
    /**
     * Apply update rule to batch of particles
     * @param {Array} attributes - Original attributes [N x F]
     * @param {Float32Array} outputMatrix - NN output matrix [numParticles * outputFeatures]
     * @param {number} numPoints - Number of particles
     * @param {number} currentFireRate - Fire rate to apply
     * @returns {Array} New attributes [N x F]
     */
    _applyUpdateRule(attributes, outputMatrix, numPoints, currentFireRate) {
        const newAttributes = new Array(numPoints);
        
        for (let i = 0; i < numPoints; i++) {
            const outputOffset = i * this.outputFeatures;
            
            // Apply fire rate mask
            const shouldUpdate = Math.random() <= currentFireRate;
            if (!shouldUpdate) {
                newAttributes[i] = [...attributes[i]];
                continue;
            }
            
            let newA;
            if (this.updateRule === 'gated') {
                // Gated update: A_new = A * gate + delta * mult
                newA = new Array(this.cellFeatures);
                
                // Extract gate, delta, mult from output
                for (let f = 0; f < this.cellFeatures; f++) {
                    const gate = 1 / (1 + Math.exp(-outputMatrix[outputOffset + f])); // sigmoid
                    const delta = Math.tanh(outputMatrix[outputOffset + this.cellFeatures + f]);
                    const mult = 1 / (1 + Math.exp(-outputMatrix[outputOffset + 2 * this.cellFeatures])); // sigmoid of last element
                    
                    newA[f] = attributes[i][f] * gate + delta * mult;
                }
            } else {
                // Original update rule: A_new = A + dA * fire_rate
                newA = new Array(this.cellFeatures);
                for (let f = 0; f < this.cellFeatures; f++) {
                    newA[f] = attributes[i][f] + outputMatrix[outputOffset + f] * currentFireRate;
                }
            }
            
            newAttributes[i] = newA;
        }
        
        return newAttributes;
    }
    
    /**
     * Apply life mask to attributes
     * @param {Array} newAttributes - Attributes to modify
     * @param {Array} prevMask - Previous life mask
     * @param {Array} newMask - New life mask
     * @param {number} numPoints - Number of particles
     */
    _applyLifeMask(newAttributes, prevMask, newMask, numPoints) {
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
     * Forward pass of the NCA model (OPTIMIZED)
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
        
        // Check if we exceed buffer capacity
        if (numPoints > this.maxParticles) {
            console.warn(`Particle count ${numPoints} exceeds buffer capacity ${this.maxParticles}. Falling back to original implementation.`);
            return this._forwardOriginal(points, volumes, attributes, h, hashGrid, fireRate, useAlpha);
        }
        
        // 1. Calculate life mask before update
        let prevMask = null;
        if (useAlpha) {
            prevMask = this._lifeMask(points, volumes, attributes, h, hashGrid);
        }
        
        // 2. Perceive - calculate gradients
        const gradients = this._perceive(points, volumes, attributes, h, hashGrid);
        
        // 3. OPTIMIZED: Batch NN processing
        const inputMatrix = this._prepareInputBatch(attributes, gradients, numPoints, h);
        const outputMatrix = this._forwardBatch(inputMatrix, numPoints);
        
        // 4. OPTIMIZED: Batch update rule application
        const newAttributes = this._applyUpdateRule(attributes, outputMatrix, numPoints, currentFireRate);
        
        // 5. Final life mask and application
        if (useAlpha) {
            const newMask = this._lifeMask(points, volumes, newAttributes, h, hashGrid);
            this._applyLifeMask(newAttributes, prevMask, newMask, numPoints);
        }
        
        return newAttributes;
    }
    
    /**
     * Original forward pass implementation (fallback)
     * @param {Array} points - Particle positions
     * @param {Array} volumes - Particle volumes
     * @param {Array} attributes - Particle attributes [N x F]
     * @param {number} h - Kernel radius
     * @param {Object} hashGrid - Hash grid for neighbor finding
     * @param {number} fireRate - Override fire rate (optional)
     * @returns {Array} Updated attributes [N x F]
     */
    _forwardOriginal(points, volumes, attributes, h, hashGrid, fireRate = null, useAlpha = true) {
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
