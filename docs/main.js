/**
 * Main application logic for SPHNCA 2D demo
 */

import { NCA } from './nca.js';
import { HashGrid } from './hashgrid.js';
import { operators } from './sph.js';
import { vec2, colorUtils, perfUtils, pointUtils } from './utils.js';

// Global configuration
const config = {
    // Simulation parameters
    h: 0.1,                    // Kernel radius
    fireRate: 0.5,              // Update probability
    gridWidth: 64,              // Number of particles horizontally
    gridHeight: 64,             // Number of particles vertically
    particleSpacing: 2. / 64,       // Spacing between particles
    
    // Canvas parameters
    canvasWidth: 1000,
    canvasHeight: 1000,
    particleSize: 0.5 * 1000. / 64,            // Particle render size
    
    // Animation parameters
    fps: 30,                    // Target FPS
    maxSteps: Infinity,             // Maximum simulation steps
    
    // Visualization parameters
    colorMode: 'rgba',          // 'rgba', 'activity'
    
    // Seed parameters
    initialSeed: 'random',     // 'random', 'radial'
    seedRadius: 0.16,
    brushRadius: 0.16,

    useAlpha: false,          // Use alpha channel in rendering
    useWrap: true,          // Wrap coordinates for periodic boundary conditions
};

// Global state
let state = {
    // Core objects (moved to worker)
    nca: null,
    hashGrid: null,
    
    // Simulation data (received from worker)
    points: [],
    attributes: [],
    volumes: [],
    wrapIndex: [],
    
    // Canvas and rendering
    canvas: null,
    ctx: null,
    
    // Animation control
    isRunning: false,
    animationId: null,
    stepCount: 0,
    lastTime: 0,
    
    // Performance tracking
    frameTime: 0,
    updateTime: 0,
    renderTime: 0,
    
    // Worker management
    inferenceWorker: null,
    workerSupported: false,
    pendingWorkerInit: false
};

/**
 * Initialize the application
 */
async function init() {
    console.log('Initializing SPHNCA demo...');
    
    try {
        // Initialize worker first
        initWorker();
        
        // Setup canvas
        setupCanvas();
        
        // Setup UI controls
        setupControls();
        
        if (state.workerSupported) {
            // Load weights and initialize worker
            const weights = await loadWeightsForWorker();
            await initializeWorkerSimulation(weights);
        } else {
            // Fallback to main thread
            await loadWeights();
            initializePoints();
            initializeStates();
        }
        
        // Start animation loop
        startAnimation();
        
        console.log('Initialization complete!');
        
    } catch (error) {
        console.error('Initialization failed:', error);
        showError(`Initialization failed: ${error.message}`);
    }
}

/**
 * Initialize worker
 */
function initWorker() {
    if (typeof Worker !== 'undefined') {
        try {
            state.inferenceWorker = new Worker('./inference-worker.js', { type: 'module' });
            state.inferenceWorker.onmessage = handleWorkerMessage;
            state.inferenceWorker.onerror = handleWorkerError;
            state.workerSupported = true;
            console.log('Inference worker initialized');
        } catch (error) {
            console.warn('Failed to initialize worker:', error);
            state.workerSupported = false;
        }
    } else {
        console.warn('Web Workers not supported, falling back to main thread');
        state.workerSupported = false;
    }
}

/**
 * Handle messages from worker
 */
function handleWorkerMessage(e) {
    const { type, data } = e.data;
    
    switch (type) {
        case 'STATE_UPDATE':
            // Update local state from worker
            updateStateFromWorker(data);
            break;
            
        default:
            console.warn('Unknown worker message type:', type);
    }
}

/**
 * Handle worker errors
 */
function handleWorkerError(error) {
    console.error('Worker error:', error);
    showError(`Worker error: ${error.message}`);
}

/**
 * Update local state from worker data
 */
function updateStateFromWorker(data) {
    // Reconstruct points array from flat buffer
    state.points = [];
    for (let i = 0; i < data.pointsLength; i++) {
        state.points.push([
            data.pointsFlat[i * 2],
            data.pointsFlat[i * 2 + 1]
        ]);
    }
    
    // Reconstruct attributes array from flat buffer
    state.attributes = [];
    for (let i = 0; i < data.pointsLength; i++) {
        const row = [];
        for (let j = 0; j < data.attributesLength; j++) {
            row.push(data.attributesFlat[i * data.attributesLength + j]);
        }
        state.attributes.push(row);
    }
    
    // Update other state
    state.stepCount = data.stepCount;
    state.updateTime = data.updateTime;
}

/**
 * Load weights for worker
 */
async function loadWeightsForWorker() {
    console.log('Loading model weights for worker...');

    // Read selection from weightSelector
    const weightSelector = document.getElementById('weightSelector');
    const selectedWeight = weightSelector.value;
    const weightFile = `weights/${selectedWeight}.json`;
    
    let response = await fetch(weightFile);
    if (!response.ok) {
        response = await fetch('weights/default.json');
        if (!response.ok) throw new Error(`Failed to load model weights: ${response.statusText}`);
    }
    const weights = await response.json();
    console.log('Model weights loaded successfully for worker');

    // Update config from weights
    config.h = weights.config.h || config.h;
    config.fireRate = weights.config.fire_rate || config.fireRate;
    config.useAlpha = weights.config.mode === 'image' ? true : false;
    config.useWrap = weights.config.mode === 'image' ? false : true;
    config.initialSeed = weights.config.mode === 'image' ? 'radial' : 'random';
    
    return weights;
}

/**
 * Initialize worker simulation
 */
async function initializeWorkerSimulation(weights) {
    if (!state.workerSupported || !state.inferenceWorker) {
        throw new Error('Worker not available');
    }
    
    // Get UI configuration
    updateConfigFromUI();
    
    state.pendingWorkerInit = true;
    
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            reject(new Error('Worker initialization timeout'));
        }, 10000);
        
        const originalHandler = state.inferenceWorker.onmessage;
        state.inferenceWorker.onmessage = (e) => {
            if (e.data.type === 'STATE_UPDATE' && state.pendingWorkerInit) {
                clearTimeout(timeout);
                state.pendingWorkerInit = false;
                state.inferenceWorker.onmessage = originalHandler;
                updateStateFromWorker(e.data.data);
                
                // Initialize NCA for rendering (needed for toRGBA)
                state.nca = new NCA(weights);
                console.log('Worker simulation initialized');
                resolve();
            } else {
                originalHandler(e);
            }
        };
        
        // Send initialization message to worker
        state.inferenceWorker.postMessage({
            type: 'INIT',
            data: { weights, config }
        });
    });
}

/**
 * Update config from UI elements
 */
function updateConfigFromUI() {
    // Resolution
    const resolutionSelector = document.getElementById('resolution');
    if (resolutionSelector) {
        const resolutionMap = {
            1: 50,
            2: 64,
            3: 80,
            4: 100,
            5: 128
        };
        const resolution = resolutionMap[resolutionSelector.value] || 64;
        config.gridWidth = resolution;
        config.gridHeight = resolution;
        config.particleSpacing = 2. / resolution;
        config.particleSize = 0.5 * config.canvasWidth / resolution;
    }
    
    // Point pattern
    const pointPatternSelector = document.getElementById('pointPattern');
    if (pointPatternSelector) {
        config.pointPattern = pointPatternSelector.value;
    }
    
    // Noise
    const noiseSelector = document.getElementById('noise');
    if (noiseSelector) {
        config.noiseAmount = parseFloat(noiseSelector.value);
    }
    
    const spatialNoiseSelector = document.getElementById('spatialNoise');
    if (spatialNoiseSelector) {
        config.spatiallyVaryingNoise = spatialNoiseSelector.checked;
    }
}

async function loadWeights() {
    // Load model weights
    console.log('Loading model weights...');

    // Read selection from weightSelector
    const weightSelector = document.getElementById('weightSelector');
    const selectedWeight = weightSelector.value;
    const weightFile = `weights/${selectedWeight}.json`;
    
    let response = await fetch(weightFile);
    if (!response.ok) {
        response = await fetch('weights/default.json');
        if (!response.ok) throw new Error(`Failed to load model weights: ${response.statusText}`);
    }
    const weights = await response.json();
    console.log('Model weights loaded successfully');
        
    // Initialize NCA model
    state.nca = new NCA(weights);
    console.log('NCA model initialized');
    console.log('Model stats:', state.nca.getStats());

    config.h = weights.config.h || config.h;
    config.fireRate = weights.config.fire_rate || config.fireRate;
    config.useAlpha = weights.config.mode === 'image' ? true : false;
    config.useWrap = weights.config.mode === 'image' ? false : true;
    config.initialSeed = weights.config.mode === 'image' ? 'radial' : 'random';
}

/**
 * Setup canvas and rendering context
 */
function setupCanvas() {
    state.canvas = document.getElementById('canvas');
    if (!state.canvas) {
        throw new Error('Canvas element not found');
    }
    
    state.canvas.width = config.canvasWidth;
    state.canvas.height = config.canvasHeight;
    state.ctx = state.canvas.getContext('2d');
    
    // Add mouse interaction
    state.canvas.addEventListener('click', handleCanvasClick);
    state.canvas.addEventListener('mousemove', handleCanvasMouseMove);
    
    console.log(`Canvas setup: ${config.canvasWidth}x${config.canvasHeight}`);
}

function createWrappedBoundary(points, worldWidth, worldHeight) {
    const wrappedPoints = [];
    const wrapIndex = [];
    const wrapMargin = config.h + config.particleSpacing;


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

    state.wrapIndex = newWrapIndex;
    return newPoints;
}

function updateWrappedBoundary(points, attributes) {
    const wrapIndex = state.wrapIndex;
    
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
 * Initialize simulation data
 */
function initializePoints() {
    // Calculate world dimensions
    const resolutionSelector = document.getElementById('resolution').value;
    const resolutionMap = {
        1: 50,
        2: 64,
        3: 80,
        4: 100,
        5: 128
    };
    const resolution = resolutionMap[resolutionSelector] || 64;
    config.gridWidth = resolution;
    config.gridHeight = resolution;
    config.particleSpacing = 2. / resolution; // Adjust particle spacing based on resolution
    config.particleSize = 0.5 * config.canvasWidth / resolution; // Adjust particle size based on canvas width

    const worldWidth = config.gridWidth * config.particleSpacing;
    const worldHeight = config.gridHeight * config.particleSpacing;
    
    const pointPatternSelector = document.getElementById('pointPattern');
    const pointPattern = pointPatternSelector ? pointPatternSelector.value : 'square';
    console.log(`Using point pattern: ${pointPattern}`);

    if (pointPattern === 'hexagonal') {
        // Create hexagonal grid
        state.points = pointUtils.createHexagonalGrid(
            worldWidth, 
            worldHeight, 
            config.particleSpacing,
            config.particleSpacing * 0.2,  // Center offset
            config.particleSpacing * 0.0
        );
    } else {
        // Create square grid
        state.points = pointUtils.createUniformGrid(
            worldWidth, 
            worldHeight, 
            config.particleSpacing,
            config.particleSpacing * 0.5,  // Center offset
            config.particleSpacing * 0.5
        );
    }
    
    // Add noise
    const noiseAmount = document.getElementById('noise').value;
    const spatiallyVaryingNoise = document.getElementById('spatialNoise').checked;
    if (noiseAmount > 0) {
        const noiseRange = noiseAmount * config.particleSpacing;
        if (spatiallyVaryingNoise) {
            state.points = pointUtils.addSpatialNoise(state.points, noiseRange, worldWidth, worldHeight);
        } else {
            state.points = pointUtils.addNoise(state.points, noiseRange);
        }
    }

    // Create wrapped boundary if enabled
    if (config.useWrap) {
        state.points = createWrappedBoundary(state.points, worldWidth, worldHeight);
    }

    // Initialize hash grid
    state.hashGrid = new HashGrid(config.h, worldWidth, worldHeight);
    state.hashGrid.build(state.points);

    // Calculate initial volumes
    state.volumes = operators.calculate_volumes(state.points, state.hashGrid, config.h);
}

function initializeStates() {
    const worldWidth = config.gridWidth * config.particleSpacing;
    const worldHeight = config.gridHeight * config.particleSpacing;

    // Initialize attributes (16 features to match model)
    const numFeatures = state.nca.cellFeatures;
    if (config.initialSeed === 'radial') {
        state.attributes = pointUtils.initializeAttributes(state.points.length, numFeatures, 0, 0);
    
        // Add seed in the center
        const centerX = worldWidth * 0.5;
        const centerY = worldHeight * 0.5;
        pointUtils.addCircularSeed(
            state.points, 
            state.attributes, 
            [centerX, centerY], 
            config.seedRadius 
        );
    } else if (config.initialSeed === 'random') {
        state.attributes = pointUtils.initializeAttributes(state.points.length, numFeatures, 0, 1);
    }
    
    // Reset step counter
    state.stepCount = 0;
    
    console.log(`Simulation initialized:`);
}

/**
 * Update simulation by one step
 */
function update() {
    const timer = perfUtils.timer();
    
    if (config.useWrap) {
        // Update wrapped boundary
        updateWrappedBoundary(state.points, state.attributes);
    }
    
    // Rebuild hash grid
    state.hashGrid.build(state.points);
    
    // Calculate volumes
    state.volumes = operators.calculate_volumes(state.points, state.hashGrid, config.h);
    
    // Run NCA forward pass
    state.attributes = state.nca.forward(
        state.points,
        state.volumes,
        state.attributes,
        config.h,
        state.hashGrid,
        config.fireRate,
        config.useAlpha
    );
    
    state.stepCount++;
    state.updateTime = timer.elapsed();
}

/**
 * Render the current state
 */
function render() {
    const timer = perfUtils.timer();
    
    // Clear canvas
    state.ctx.fillStyle = '#000000';
    state.ctx.fillRect(0, 0, config.canvasWidth, config.canvasHeight);
    
    // Convert world coordinates to screen coordinates
    const worldWidth = config.gridWidth * config.particleSpacing;
    const worldHeight = config.gridHeight * config.particleSpacing;
    const scaleX = config.canvasWidth / worldWidth;
    const scaleY = config.canvasHeight / worldHeight;
    
    // Render particles
    const rgba = state.nca.toRGBA(state.attributes);
    
    for (let i = 0; i < state.points.length; i++) {
        const point = state.points[i];
        let color = rgba[i];

        // Skip rendering out of bounds particles
        if (point[0] < -config.particleSpacing || point[0] > worldWidth + config.particleSpacing ||
            point[1] < -config.particleSpacing || point[1] > worldHeight + config.particleSpacing) {
            continue;
        }

        // Use wrap index if enabled
        // if (config.useWrap && state.wrapIndex[i] !== i)
        //     color = rgba[state.wrapIndex[i]];
        
        // Convert to screen coordinates
        const screenX = (worldWidth - point[1] + config.particleSpacing * 0.5) * scaleX;
        const screenY = (worldHeight - point[0] + config.particleSpacing * 0.5) * scaleY;
        
        // Skip if alpha is too low
        if (config.useAlpha && color[3] < 0.1) continue;
        
        // Set color based on mode
        let fillStyle;
        const activity = config.useAlpha ? color[3] : 1;
        switch (config.colorMode) {
            case 'rgba':
                fillStyle = colorUtils.rgbaToString(color[0], color[1], color[2], activity);
                break;
            case 'activity':
                fillStyle = colorUtils.rgbaToString(activity, activity, activity, 1);
                break;
            default:
                fillStyle = colorUtils.rgbaToString(color[0], color[1], color[2], color[3]);
        }
        
        // Draw particle
        state.ctx.fillStyle = fillStyle;
        state.ctx.beginPath();
        state.ctx.arc(screenX, screenY, config.particleSize, 0, 2 * Math.PI);
        state.ctx.fill();
    }
    
    // Render UI overlay
    renderUI();
    
    state.renderTime = timer.elapsed();
}

/**
 * Render UI overlay with stats
 */
function renderUI() {
    state.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    state.ctx.fillRect(10, 10, 200, 120);
    
    state.ctx.fillStyle = '#ffffff';
    state.ctx.font = '12px monospace';
    
    const stats = [
        `Step: ${state.stepCount}`,
        `FPS: ${(1000 / state.frameTime).toFixed(1)}`,
        `Update: ${state.updateTime.toFixed(1)}ms`,
        `Render: ${state.renderTime.toFixed(1)}ms`,
        `Particles: ${state.points.length}`,
        `Status: ${state.isRunning ? 'Running' : 'Paused'}`
    ];
    
    for (let i = 0; i < stats.length; i++) {
        state.ctx.fillText(stats[i], 15, 25 + i * 15);
    }
}

/**
 * Animation loop
 */
function animate(currentTime) {
    state.frameTime = currentTime - state.lastTime;
    
    // Limit frame rate
    const targetFrameTime = 1000 / config.fps;
    if (state.frameTime >= targetFrameTime) {
        state.lastTime = currentTime;
        render();
        
        // Only run update if using main thread (fallback mode)
        if (!state.workerSupported && state.isRunning && state.stepCount < config.maxSteps) {
            update();
        }
    }
    
    state.animationId = requestAnimationFrame(animate);
}

/**
 * Start animation
 */
function startAnimation() {
    if (!state.animationId) {
        state.lastTime = performance.now();
        state.animationId = requestAnimationFrame(animate);
    }
    state.isRunning = true;
    
    // Start worker updates if using worker
    if (state.workerSupported && state.inferenceWorker) {
        state.inferenceWorker.postMessage({ type: 'START' });
    }
}

/**
 * Stop animation
 */
function stopAnimation() {
    state.isRunning = false;
    
    // Stop worker updates if using worker
    if (state.workerSupported && state.inferenceWorker) {
        state.inferenceWorker.postMessage({ type: 'STOP' });
    }
}

/**
 * Reset simulation
 */
function reset() {
    stopAnimation();
    
    if (state.workerSupported && state.inferenceWorker) {
        // Reset worker simulation
        state.inferenceWorker.postMessage({ type: 'RESET' });
    } else {
        // Reset main thread simulation
        initializeStates();
    }
    
    startAnimation();
}

/**
 * Handle canvas click events
 */
function handleCanvasClick(event) {
    const rect = state.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Convert screen coordinates to world coordinates
    const worldWidth = config.gridWidth * config.particleSpacing;
    const worldHeight = config.gridHeight * config.particleSpacing;
    const worldX = (1. - y / rect.width) * worldWidth;
    const worldY = (1. - x / rect.height) * worldHeight;

    const brushRadiusElement = document.getElementById('brushRadius');
    const brushTypeElement = document.getElementById('brushType');
    const brushRadius = brushRadiusElement ? parseFloat(brushRadiusElement.value) : config.brushRadius;
    const brushType = brushTypeElement ? brushTypeElement.value : 'seed';
    
    if (state.workerSupported && state.inferenceWorker) {
        // Send interaction to worker
        state.inferenceWorker.postMessage({
            type: 'USER_INTERACTION',
            data: {
                type: 'CANVAS_CLICK',
                worldX,
                worldY,
                brushRadius,
                brushType
            }
        });
    } else {
        // Handle directly on main thread
        pointUtils.addCircularSeed(
            state.points,
            state.attributes,
            [worldX, worldY],
            brushRadius,
            brushType === 'randomize'
        );
    }
}

/**
 * Handle canvas mouse move events
 */
function handleCanvasMouseMove(event) {
    // Could be used for interactive painting in the future
}

/**
 * Setup UI controls
 */
function setupControls() {
    // Weight load and reset button
    const loadWeightsBtn = document.getElementById('loadWeights');
    if (loadWeightsBtn) {
        loadWeightsBtn.addEventListener('click', async () => {
            try {
                stopAnimation();
                
                if (state.workerSupported && state.inferenceWorker) {
                    // Reload weights and reinitialize worker
                    const weights = await loadWeightsForWorker();
                    await initializeWorkerSimulation(weights);
                } else {
                    // Fallback to main thread
                    await loadWeights();
                    initializePoints();
                    initializeStates();
                }
                
                startAnimation();
                console.log('Weights loaded and simulation reset');
            } catch (error) {
                console.error('Failed to load weights:', error);
                showError(`Failed to load weights: ${error.message}`);
            }
        });
    }

    // Point pattern selector
    const pointPatternBtn = document.getElementById('applyPointPattern');
    if (pointPatternBtn) {
        pointPatternBtn.addEventListener('click', async () => {
            try {
                stopAnimation();
                
                if (state.workerSupported && state.inferenceWorker) {
                    // Update config from UI and send to worker
                    updateConfigFromUI();
                    
                    // Send message to worker to reinitialize points only
                    state.inferenceWorker.postMessage({
                        type: 'REINIT_POINTS',
                        data: { config }
                    });
                } else {
                    // Fallback to main thread
                    initializePoints();
                    initializeStates();
                }
                
                startAnimation();
                console.log('Point pattern applied and simulation reset');
            } catch (error) {
                console.error('Failed to apply point pattern:', error);
                showError(`Failed to apply point pattern: ${error.message}`);
            }
        });
    }

    // Play/Pause button
    const playPauseBtn = document.getElementById('playPause');
    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', () => {
            if (state.isRunning) {
                stopAnimation();
                playPauseBtn.textContent = 'Play';
            } else {
                startAnimation();
                playPauseBtn.textContent = 'Pause';
            }
        });
    }
    
    // Reset button
    const resetBtn = document.getElementById('reset');
    if (resetBtn) {
        resetBtn.addEventListener('click', reset);
    }
    
    // Color mode selector
    const colorModeSelect = document.getElementById('colorMode');
    if (colorModeSelect) {
        colorModeSelect.value = config.colorMode;
        colorModeSelect.addEventListener('change', (e) => {
            config.colorMode = e.target.value;
        });
    }
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.getElementById('error');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    } else {
        alert(message);
    }
}

// Export functions for global access
window.sphnca = {
    init,
    reset,
    startAnimation,
    stopAnimation,
    config,
    state
};

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
