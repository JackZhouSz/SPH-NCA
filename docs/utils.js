/**
 * Utility functions for vector math and array operations
 */

// 2D Vector operations
export const vec2 = {
    // Create a new vector
    create: (x = 0, y = 0) => [x, y],
    
    // Add two vectors
    add: (a, b) => [a[0] + b[0], a[1] + b[1]],
    
    // Subtract two vectors
    subtract: (a, b) => [a[0] - b[0], a[1] - b[1]],
    
    // Scale a vector by a scalar
    scale: (v, s) => [v[0] * s, v[1] * s],
    
    // Dot product of two vectors
    dot: (a, b) => a[0] * b[0] + a[1] * b[1],
    
    // Squared magnitude of a vector
    magnitudeSq: (v) => v[0] * v[0] + v[1] * v[1],
    
    // Magnitude of a vector
    magnitude: (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1]),
    
    // Normalize a vector
    normalize: (v) => {
        const mag = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
        return mag > 0 ? [v[0] / mag, v[1] / mag] : [0, 0];
    },
    
    // Distance between two points
    distance: (a, b) => {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        return Math.sqrt(dx * dx + dy * dy);
    },
    
    // Squared distance between two points
    distanceSq: (a, b) => {
        const dx = a[0] - b[0];
        const dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }
};

// Array utilities
export const arrayUtils = {
    // Create a 2D array filled with a value
    create2D: (width, height, fillValue = 0) => {
        return Array(height).fill().map(() => Array(width).fill(fillValue));
    },
    
    // Flatten a 2D array
    flatten: (arr2d) => {
        return arr2d.reduce((flat, row) => flat.concat(row), []);
    },
    
    // Create an array of zeros
    zeros: (length) => new Array(length).fill(0),
    
    // Create an array with random values
    random: (length, min = 0, max = 1) => {
        return Array(length).fill().map(() => Math.random() * (max - min) + min);
    },
    
    // Clamp a value between min and max
    clamp: (value, min, max) => Math.max(min, Math.min(max, value)),
    
    // Linear interpolation
    lerp: (a, b, t) => a + (b - a) * t
};

// Math utilities
export const mathUtils = {
    // Generate random number between min and max
    random: (min = 0, max = 1) => Math.random() * (max - min) + min,
    
    // Generate random integer between min and max (inclusive)
    randomInt: (min, max) => Math.floor(Math.random() * (max - min + 1)) + min,
    
    // Clamp value between min and max
    clamp: (value, min, max) => Math.max(min, Math.min(max, value)),
    
    // Linear interpolation
    lerp: (a, b, t) => a + (b - a) * t,
    
    // Smooth step function
    smoothstep: (edge0, edge1, x) => {
        const t = mathUtils.clamp((x - edge0) / (edge1 - edge0), 0, 1);
        return t * t * (3 - 2 * t);
    },
    
    // Map value from one range to another
    map: (value, inMin, inMax, outMin, outMax) => {
        return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
    }
};

// Color utilities
export const colorUtils = {
    // Convert RGBA values to CSS color string
    rgbaToString: (r, g, b, a = 1) => {
        const ri = Math.round(mathUtils.clamp(r, 0, 1) * 255);
        const gi = Math.round(mathUtils.clamp(g, 0, 1) * 255);
        const bi = Math.round(mathUtils.clamp(b, 0, 1) * 255);
        const ai = mathUtils.clamp(a, 0, 1);
        return `rgba(${ri}, ${gi}, ${bi}, ${ai})`;
    },
    
    // Convert HSV to RGB
    hsvToRgb: (h, s, v) => {
        const c = v * s;
        const x = c * (1 - Math.abs((h / 60) % 2 - 1));
        const m = v - c;
        
        let r, g, b;
        if (h >= 0 && h < 60) {
            [r, g, b] = [c, x, 0];
        } else if (h >= 60 && h < 120) {
            [r, g, b] = [x, c, 0];
        } else if (h >= 120 && h < 180) {
            [r, g, b] = [0, c, x];
        } else if (h >= 180 && h < 240) {
            [r, g, b] = [0, x, c];
        } else if (h >= 240 && h < 300) {
            [r, g, b] = [x, 0, c];
        } else {
            [r, g, b] = [c, 0, x];
        }
        
        return [r + m, g + m, b + m];
    }
};

// Performance utilities
export const perfUtils = {
    // Simple timer for performance measurement
    timer: () => {
        const start = performance.now();
        return {
            elapsed: () => performance.now() - start,
            log: (label = 'Timer') => console.log(`${label}: ${(performance.now() - start).toFixed(2)}ms`)
        };
    },
    
    // Throttle function calls
    throttle: (func, delay) => {
        let timeoutId;
        let lastExecTime = 0;
        return function (...args) {
            const currentTime = Date.now();
            
            if (currentTime - lastExecTime > delay) {
                func.apply(this, args);
                lastExecTime = currentTime;
            } else {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    func.apply(this, args);
                    lastExecTime = Date.now();
                }, delay - (currentTime - lastExecTime));
            }
        };
    }
};

export const pointUtils = {
    /**
     * Create a uniform grid of particles
     * @param {number} width - Grid width
     * @param {number} height - Grid height
     * @param {number} spacing - Particle spacing
     * @param {number} offsetX - X offset
     * @param {number} offsetY - Y offset
     * @returns {Array} Array of particle positions
     */
    createUniformGrid: (width, height, spacing, offsetX = 0, offsetY = 0) => {
        const points = [];
        const nx = Math.floor(width / spacing);
        const ny = Math.floor(height / spacing);
        
        for (let j = 0; j < ny; j++) {
            for (let i = 0; i < nx; i++) {
                const x = offsetX + (i + 0.5) * spacing;
                const y = offsetY + (j + 0.5) * spacing;
                points.push([x, y]);
            }
        }
        
        return points;
    },

    /**
     * Create a hexagonal grid of particles
     * @param {number} width - Grid width
     * @param {number} height - Grid height
     * @param {number} spacing - Particle spacing (horizontal distance between particles)
     * @param {number} offsetX - X offset
     * @param {number} offsetY - Y offset
     * @returns {Array} Array of particle positions
     */
    createHexagonalGrid: (width, height, spacing, offsetX = 0, offsetY = 0) => {
        const points = [];
        
        // For hexagonal grid, vertical spacing is different from horizontal
        const verticalSpacing = spacing * Math.sqrt(3) / 2;
        
        // Calculate number of rows and approximate columns
        const ny = Math.ceil(height / verticalSpacing);
        const nx = Math.ceil(width / spacing);
        
        for (let j = 0; j < ny; j++) {
            // Alternate rows are offset by half spacing
            const isOddRow = j % 2 === 1;
            const rowOffsetX = isOddRow ? spacing / 2 : 0;
            
            // For odd rows, we might fit one less particle due to the offset
            const rowNx = isOddRow ? Math.floor((width - spacing / 2) / spacing) : nx;
            
            for (let i = 0; i < rowNx; i++) {
                const x = offsetX + rowOffsetX + (i + 0.5) * spacing;
                const y = offsetY + (j + 0.5) * verticalSpacing;
                if (x >= 0 && x <= width && y >= 0 && y <= height)
                    points.push([x, y]);
            }
        }
        
        return points;
    },
    
    /**
     * Add random noise to particle positions
     * @param {Array} points - Array of particle positions
     * @param {number} amount - Noise amount (fraction of spacing)
     * @param {number} spacing - Original particle spacing
     * @returns {Array} Array of noisy particle positions
     */
    addNoise: (points, amount) => {
        const noisyPoints = [];
        const noiseRange = amount;
        
        for (const point of points) {
            const noiseX = (Math.random() - 0.5) * noiseRange;
            const noiseY = (Math.random() - 0.5) * noiseRange;
            noisyPoints.push([
                point[0] + noiseX,
                point[1] + noiseY
            ]);
        }
        
        return noisyPoints;
    },

    addSpatialNoise: (points, amount, worldWidth, worldHeight) => {
        const noisyPoints = [];
        const noiseRange = amount;
        for (const point of points) {
            const spatialFactor = 0.5 * (point[0] / worldWidth + point[1] / worldHeight);
            const noiseX = (Math.random() - 0.5) * noiseRange * spatialFactor;
            const noiseY = (Math.random() - 0.5) * noiseRange * spatialFactor;
            noisyPoints.push([
                point[0] + noiseX,
                point[1] + noiseY
            ]);
        }
        
        return noisyPoints;
    },
    
    /**
     * Initialize particle attributes with random values
     * @param {number} numPoints - Number of particles
     * @param {number} numFeatures - Number of features per particle
     * @param {number} min - Minimum value
     * @param {number} max - Maximum value
     * @returns {Array} Array of particle attributes [N x F]
     */
    initializeAttributes: (numPoints, numFeatures, min = 0, max = 1) => {
        const attributes = new Array(numPoints);
        
        for (let i = 0; i < numPoints; i++) {
            attributes[i] = new Array(numFeatures);
            for (let f = 0; f < numFeatures; f++) {
                attributes[i][f] = Math.random() * (max - min) + min;
            }
        }
        
        return attributes;
    },
    
    /**
     * Add a circular seed pattern to attributes
     * @param {Array} points - Array of particle positions
     * @param {Array} attributes - Array of particle attributes
     * @param {Array} center - Center position [x, y]
     * @param {number} radius - Seed radius
     */
    addCircularSeed: (points, attributes, center, radius, randomize = false) => {
        const radiusSq = radius * radius;
        
        for (let i = 0; i < points.length; i++) {
            const distSq = vec2.distanceSq(points[i], center);
            if (distSq <= radiusSq) {
                // Smooth falloff from center
                const dist = Math.sqrt(distSq);
                const weight = Math.pow(1 - dist / radius, 3);
                
                for (let f = 0; f < attributes[i].length; f++) {
                    if (randomize) {
                        // Randomize the seed value
                        attributes[i][f] *= 1. - weight;
                        attributes[i][f] += Math.random() * weight;
                    } else {
                        // Add a fixed value
                        attributes[i][f] += weight;
                    }
                }
            }
        }
    }
};
