/**
 * Inference Worker for SPHNCA 2D demo
 * Handles heavy computation off the main thread
 */

import { NCA } from './nca.js';
import { HashGrid } from './hashgrid.js';
import { operators } from './sph.js';
import { vec2, colorUtils, perfUtils, pointUtils } from './utils.js';

// Worker state
let workerState = {
    // Core objects
    nca: null,
    hashGrid: null,
    
    // Simulation data
    points: [],
    attributes: [],
    volumes: [],
    wrapIndex: [],
    
    // Configuration
    config: null,
    
    // Animation control
    isRunning: false,
    stepCount: 0,
    updateInterval: null,
    
    // Performance tracking
    updateTime: 0
};

/**
 * Initialize the simulation in worker
 */
async function initializeSimulation(weights, config) {
    console.log('Worker: Initializing simulation...');
    
    // Store config
    workerState.config = { ...config };
    
    // Initialize NCA model
    workerState.nca = new NCA(weights);
    console.log('Worker: NCA model initialized');
    
    // Update config from weights
    workerState.config.h = weights.config.h || config.h;
    workerState.config.fireRate = weights.config.fire_rate || config.fireRate;
    workerState.config.useAlpha = weights.config.mode === 'image' ? true : false;
    workerState.config.useWrap = weights.config.mode === 'image' ? false : true;
    workerState.config.initialSeed = weights.config.mode === 'image' ? 'radial' : 'random';
    
    // Initialize points and states
    initializePoints();
    initializeStates();
    
    console.log('Worker: Simulation initialized');
    
    // Send initial state to main thread
    sendStateToMain();
}

/**
 * Initialize simulation points
 */
function initializePoints() {
    const config = workerState.config;
    
    // Calculate world dimensions
    const worldWidth = config.gridWidth * config.particleSpacing;
    const worldHeight = config.gridHeight * config.particleSpacing;
    
    // Create grid based on point pattern
    if (config.pointPattern === 'hexagonal') {
        workerState.points = pointUtils.createHexagonalGrid(
            worldWidth, 
            worldHeight, 
            config.particleSpacing,
            config.particleSpacing * 0.2,
            config.particleSpacing * 0.0
        );
    } else {
        workerState.points = pointUtils.createUniformGrid(
            worldWidth, 
            worldHeight, 
            config.particleSpacing,
            config.particleSpacing * 0.5,
            config.particleSpacing * 0.5
        );
    }
    
    // Add noise if specified
    if (config.noiseAmount > 0) {
        const noiseRange = config.noiseAmount * config.particleSpacing;
        if (config.spatiallyVaryingNoise) {
            workerState.points = pointUtils.addSpatialNoise(workerState.points, noiseRange, worldWidth, worldHeight);
        } else {
            workerState.points = pointUtils.addNoise(workerState.points, noiseRange);
        }
    }

    // Create wrapped boundary if enabled
    if (config.useWrap) {
        workerState.points = createWrappedBoundary(workerState.points, worldWidth, worldHeight);
    }

    // Initialize hash grid
    workerState.hashGrid = new HashGrid(config.h, worldWidth, worldHeight);
    workerState.hashGrid.build(workerState.points);

    // Calculate initial volumes
    workerState.volumes = operators.calculate_volumes(workerState.points, workerState.hashGrid, config.h);
}

/**
 * Initialize simulation states
 */
function initializeStates() {
    const config = workerState.config;
    const worldWidth = config.gridWidth * config.particleSpacing;
    const worldHeight = config.gridHeight * config.particleSpacing;

    // Initialize attributes
    const numFeatures = workerState.nca.cellFeatures;
    if (config.initialSeed === 'radial') {
        workerState.attributes = pointUtils.initializeAttributes(workerState.points.length, numFeatures, 0, 0);
    
        // Add seed in the center
        const centerX = worldWidth * 0.5;
        const centerY = worldHeight * 0.5;
        pointUtils.addCircularSeed(
            workerState.points, 
            workerState.attributes, 
            [centerX, centerY], 
            config.seedRadius 
        );
    } else if (config.initialSeed === 'random') {
        workerState.attributes = pointUtils.initializeAttributes(workerState.points.length, numFeatures, 0, 1);
    }
    
    // Reset step counter
    workerState.stepCount = 0;
}

/**
 * Create wrapped boundary for periodic conditions
 */
function createWrappedBoundary(points, worldWidth, worldHeight) {
    const wrappedPoints = [];
    const wrapIndex = [];
    const wrapMargin = workerState.config.h + workerState.config.particleSpacing;

    for (let i = 0; i < points.length; i++) {
        const point = points[i];
        let x = point[0];
        let y = point[1];
        
        // Which 9 regions does this point belong to?
        const xbound = x < wrapMargin? -1 : x > worldWidth - wrapMargin? 1 : 0;
        const ybound = y < wrapMargin? -1 : y > worldHeight - wrapMargin? 1 : 0;
        
        if (xbound < 0) {
            wrappedPoints.push([x + worldWidth, y]);
            wrapIndex.push(i);
        } else if (xbound > 0) {
            wrappedPoints.push([x - worldWidth, y]);
            wrapIndex.push(i);
        }
        
        if (ybound < 0) {
            wrappedPoints.push([x, y + worldHeight]);
            wrapIndex.push(i);
        } else if (ybound > 0) {
            wrappedPoints.push([x, y - worldHeight]);
            wrapIndex.push(i);
        }
        
        if (xbound < 0 && ybound < 0) {
            wrappedPoints.push([x + worldWidth, y + worldHeight]);
            wrapIndex.push(i);
        } else if (xbound < 0 && ybound > 0) {
            wrappedPoints.push([x + worldWidth, y - worldHeight]);
            wrapIndex.push(i);
        } else if (xbound > 0 && ybound < 0) {
            wrappedPoints.push([x - worldWidth, y + worldHeight]);
            wrapIndex.push(i);
        } else if (xbound > 0 && ybound > 0) {
            wrappedPoints.push([x - worldWidth, y - worldHeight]);
            wrapIndex.push(i);
        }
    }

    let newPoints = points.slice();
    newPoints.push(...wrappedPoints);

    let newWrapIndex = points.map((_, i) => i);
    newWrapIndex.push(...wrapIndex);

    workerState.wrapIndex = newWrapIndex;
    return newPoints;
}

/**
 * Update wrapped boundary
 */
function updateWrappedBoundary(points, attributes) {
    const wrapIndex = workerState.wrapIndex;
    
    for (let i = 0; i < points.length; i++) {
        const wrapIdx = wrapIndex[i];
        if (wrapIdx != i) {
            // Copy attributes from the original point
            for (let f = 0; f < attributes[wrapIdx].length; f++) {
                attributes[i][f] = attributes[wrapIdx][f];
            }
        }
    }
}

/**
 * Update simulation by one step
 */
function update() {
    const timer = perfUtils.timer();
    
    if (workerState.config.useWrap) {
        // Update wrapped boundary
        updateWrappedBoundary(workerState.points, workerState.attributes);
    }
    
    // Rebuild hash grid
    workerState.hashGrid.build(workerState.points);
    
    // Calculate volumes
    workerState.volumes = operators.calculate_volumes(workerState.points, workerState.hashGrid, workerState.config.h);
    
    // Run NCA forward pass
    workerState.attributes = workerState.nca.forward(
        workerState.points,
        workerState.volumes,
        workerState.attributes,
        workerState.config.h,
        workerState.hashGrid,
        workerState.config.fireRate,
        workerState.config.useAlpha
    );
    
    workerState.stepCount++;
    workerState.updateTime = timer.elapsed();
    
    // Send updated state to main thread
    sendStateToMain();
}

/**
 * Send current state to main thread
 */
function sendStateToMain() {
    // Create transferable arrays for efficient transfer
    const pointsFlat = new Float32Array(workerState.points.length * 2);
    for (let i = 0; i < workerState.points.length; i++) {
        pointsFlat[i * 2] = workerState.points[i][0];
        pointsFlat[i * 2 + 1] = workerState.points[i][1];
    }
    
    const attributesFlat = new Float32Array(workerState.attributes.length * workerState.attributes[0].length);
    for (let i = 0; i < workerState.attributes.length; i++) {
        for (let j = 0; j < workerState.attributes[i].length; j++) {
            attributesFlat[i * workerState.attributes[i].length + j] = workerState.attributes[i][j];
        }
    }
    
    self.postMessage({
        type: 'STATE_UPDATE',
        data: {
            pointsFlat,
            attributesFlat,
            pointsLength: workerState.points.length,
            attributesLength: workerState.attributes[0].length,
            stepCount: workerState.stepCount,
            updateTime: workerState.updateTime
        }
    }, [pointsFlat.buffer, attributesFlat.buffer]);
}

/**
 * Start continuous updates
 */
function startUpdates() {
    if (workerState.updateInterval) {
        clearInterval(workerState.updateInterval);
    }
    
    workerState.isRunning = true;
    workerState.updateInterval = setInterval(() => {
        if (workerState.isRunning) {
            update();
        }
    }, 1000 / 30); // 30 FPS updates
}

/**
 * Stop updates
 */
function stopUpdates() {
    workerState.isRunning = false;
    if (workerState.updateInterval) {
        clearInterval(workerState.updateInterval);
        workerState.updateInterval = null;
    }
}

/**
 * Handle user interactions
 */
function handleUserInteraction(data) {
    switch (data.type) {
        case 'CANVAS_CLICK':
            pointUtils.addCircularSeed(
                workerState.points,
                workerState.attributes,
                [data.worldX, data.worldY],
                data.brushRadius,
                data.brushType === 'randomize'
            );
            sendStateToMain();
            break;
            
        case 'CONFIG_UPDATE':
            // Update configuration
            Object.assign(workerState.config, data.config);
            break;
    }
}

/**
 * Reset simulation
 */
function resetSimulation() {
    initializeStates();
    sendStateToMain();
}

/**
 * Reinitialize points only (for point pattern changes)
 */
function reinitializePoints(newConfig) {
    // Update config
    Object.assign(workerState.config, newConfig);
    
    // Reinitialize points with new pattern
    initializePoints();
    
    // Reinitialize states with new points
    initializeStates();
    
    // Send updated state to main thread
    sendStateToMain();
}

/**
 * Message handler
 */
self.onmessage = function(e) {
    const { type, data } = e.data;
    
    switch (type) {
        case 'INIT':
            initializeSimulation(data.weights, data.config);
            break;
            
        case 'START':
            startUpdates();
            break;
            
        case 'STOP':
            stopUpdates();
            break;
            
        case 'RESET':
            resetSimulation();
            break;
            
        case 'REINIT_POINTS':
            reinitializePoints(data.config);
            break;
            
        case 'USER_INTERACTION':
            handleUserInteraction(data);
            break;
            
        case 'CONFIG_UPDATE':
            handleUserInteraction({ type: 'CONFIG_UPDATE', config: data });
            break;
            
        default:
            console.warn('Worker: Unknown message type:', type);
    }
};

console.log('Inference worker loaded');
