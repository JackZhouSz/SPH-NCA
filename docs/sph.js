/**
 * SPH (Smoothed Particle Hydrodynamics) kernels and operators
 */

import { vec2 } from './utils.js';

// SPH Kernels
export const kernels = {
    /**
     * Poly6 smoothing kernel for volume calculation
     * W(r, h) = (315 / (64π * h^9)) * (h² - r²)³ for r < h, 0 otherwise
     */
    smoothing_poly6: (r_sq, h) => {
        const h_sq = h * h;
        if (r_sq >= h_sq) return 0;
        
        const diff = h_sq - r_sq;
        return diff * diff * diff;
    },
    
    /**
     * Poly6 kernel normalization constant for 2D
     */
    smoothing_poly6_normalization: (h) => {
        return 315. / (64. * Math.PI * Math.pow(h, 9));
    },
    
    /**
     * Spiky kernel gradient for gradient calculation
     * ∇W(r, h) = -(45 / (π * h^6)) * (h - |r|)² * (r / |r|) for |r| < h, 0 otherwise
     */
    gradient_spiky_derivative: (r, h) => {
        const r_mag = vec2.magnitude(r);
        if (r_mag >= h || r_mag === 0) return [0, 0];
        
        const h_minus_r = h - r_mag;
        const factor = 3.0 * h_minus_r * h_minus_r / r_mag;
        
        return [-factor * r[0], -factor * r[1]];
    },
    
    /**
     * Spiky kernel normalization constant for 2D
     */
    gradient_spiky_normalization: (h) => {
        return 15.0 / (Math.PI * Math.pow(h, 6));
    }
};

// SPH Operators
export const operators = {
    /**
     * Calculate particle volumes (inverse density)
     * @param {Array} points - Array of particle positions [[x, y], ...]
     * @param {Object} hashGrid - Hash grid for neighbor finding
     * @param {number} h - Kernel radius
     * @returns {Array} Array of volume values
     */
    calculate_volumes: (points, hashGrid, h) => {
        const volumes = new Array(points.length);
        const normalization = kernels.smoothing_poly6_normalization(h);
        
        for (let i = 0; i < points.length; i++) {
            const neighbors = hashGrid.query(i, points);
            let inv_volume = 0;
            
            for (const j of neighbors) {
                const r = vec2.subtract(points[j], points[i]);
                const r_sq = vec2.magnitudeSq(r);
                inv_volume += kernels.smoothing_poly6(r_sq, h);
            }
            
            // Volume is inverse of density
            volumes[i] = 1.0 / (normalization * inv_volume + 1e-8);
        }
        
        return volumes;
    },
    
    /**
     * Calculate gradient of attributes using SPH
     * @param {Array} points - Array of particle positions
     * @param {Array} attributes - Array of particle attributes [N x F]
     * @param {Array} volumes - Array of particle volumes
     * @param {Object} hashGrid - Hash grid for neighbor finding
     * @param {number} h - Kernel radius
     * @returns {Array} Array of gradients [N x F x 2]
     */
    calculate_gradient: (points, attributes, volumes, hashGrid, h) => {
        const numPoints = points.length;
        const numFeatures = attributes[0].length;
        const gradients = new Array(numPoints);
        const normalization = kernels.gradient_spiky_normalization(h);
        
        // Initialize gradient arrays
        for (let i = 0; i < numPoints; i++) {
            gradients[i] = new Array(numFeatures);
            for (let f = 0; f < numFeatures; f++) {
                gradients[i][f] = [0, 0];
            }
        }
        
        for (let i = 0; i < numPoints; i++) {
            const neighbors = hashGrid.query(i, points);
            
            for (const j of neighbors) {
                if (i === j) continue;
                
                const r = vec2.subtract(points[j], points[i]);
                const grad_w = kernels.gradient_spiky_derivative(r, h);
                const vj = volumes[j];
                
                for (let f = 0; f < numFeatures; f++) {
                    const dA = attributes[j][f] - attributes[i][f];
                    gradients[i][f][0] += dA * grad_w[0] * vj;
                    gradients[i][f][1] += dA * grad_w[1] * vj;
                }
            }
            
            // Apply normalization
            for (let f = 0; f < numFeatures; f++) {
                gradients[i][f][0] *= normalization;
                gradients[i][f][1] *= normalization;
            }
        }
        
        return gradients;
    },
    
    /**
     * Apply blur operation using SPH smoothing
     * @param {Array} points - Array of particle positions
     * @param {Array} attributes - Array of particle attributes
     * @param {Array} volumes - Array of particle volumes
     * @param {Object} hashGrid - Hash grid for neighbor finding
     * @param {number} h - Kernel radius
     * @returns {Array} Array of blurred attributes
     */
    apply_blur: (points, attributes, volumes, hashGrid, h) => {
        const numPoints = points.length;
        const numFeatures = attributes[0].length;
        const blurred = new Array(numPoints);
        const normalization = kernels.smoothing_poly6_normalization(h);
        
        // Initialize blurred arrays
        for (let i = 0; i < numPoints; i++) {
            blurred[i] = new Array(numFeatures).fill(0);
        }
        
        for (let i = 0; i < numPoints; i++) {
            const neighbors = hashGrid.query(i, points);
            
            for (const j of neighbors) {
                const r = vec2.subtract(points[j], points[i]);
                const r_sq = vec2.magnitudeSq(r);
                const w = kernels.smoothing_poly6(r_sq, h);
                const wvj = w * volumes[j];
                
                for (let f = 0; f < numFeatures; f++) {
                    blurred[i][f] += attributes[j][f] * wvj;
                }
            }
            
            // Apply normalization
            for (let f = 0; f < numFeatures; f++) {
                blurred[i][f] *= normalization;
            }
        }
        
        return blurred;
    },
    
    /**
     * Calculate particle count in neighborhood (for debugging)
     * @param {Array} points - Array of particle positions
     * @param {Object} hashGrid - Hash grid for neighbor finding
     * @param {number} h - Kernel radius
     * @returns {Array} Array of neighbor counts
     */
    calculate_neighbor_count: (points, hashGrid, h) => {
        const counts = new Array(points.length);
        
        for (let i = 0; i < points.length; i++) {
            const neighbors = hashGrid.query(i, points);
            counts[i] = neighbors.length;
        }
        
        return counts;
    }
};
