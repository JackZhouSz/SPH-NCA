/**
 * Hash Grid implementation for efficient spatial partitioning and neighbor finding
 */

import { vec2 } from './utils.js';

export class HashGrid {
    constructor(h, width, height) {
        this.h = h;  // Cell size (kernel radius)
        this.width = width;
        this.height = height;
        
        // Calculate grid dimensions
        this.gridWidth = Math.ceil(width / h);
        this.gridHeight = Math.ceil(height / h);
        this.totalCells = this.gridWidth * this.gridHeight;
        
        // Grid storage
        this.cells = new Array(this.totalCells);
        this.clear();
    }
    
    /**
     * Clear all cells
     */
    clear() {
        for (let i = 0; i < this.totalCells; i++) {
            this.cells[i] = [];
        }
    }
    
    /**
     * Get grid coordinates from world position
     */
    getGridCoords(x, y) {
        const gx = Math.floor(x / this.h);
        const gy = Math.floor(y / this.h);
        return [gx, gy];
    }
    
    /**
     * Get cell index from grid coordinates
     */
    getCellIndex(gx, gy) {
        // Wrap coordinates to handle boundaries
        const wrappedGx = ((gx % this.gridWidth) + this.gridWidth) % this.gridWidth;
        const wrappedGy = ((gy % this.gridHeight) + this.gridHeight) % this.gridHeight;
        return wrappedGy * this.gridWidth + wrappedGx;
    }
    
    /**
     * Get cell index from world position
     */
    getCellIndexFromPos(x, y) {
        const [gx, gy] = this.getGridCoords(x, y);
        return this.getCellIndex(gx, gy);
    }
    
    /**
     * Build the hash grid from an array of points
     * @param {Array} points - Array of points, each point is [x, y]
     */
    build(points) {
        this.clear();
        
        // Insert each point into its corresponding cell
        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            const cellIndex = this.getCellIndexFromPos(point[0], point[1]);
            this.cells[cellIndex].push(i);
        }
    }
    
    /**
     * Query neighbors within radius h of a given point
     * @param {number} pointIndex - Index of the query point
     * @param {Array} points - Array of all points
     * @returns {Array} Array of neighbor point indices
     */
    query(pointIndex, points) {
        const point = points[pointIndex];
        const [gx, gy] = this.getGridCoords(point[0], point[1]);
        
        const neighbors = [];
        
        // Check 3x3 grid of cells around the point
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                const cellIndex = this.getCellIndex(gx + dx, gy + dy);
                const cellPoints = this.cells[cellIndex];
                
                // Check all points in this cell
                for (let i = 0; i < cellPoints.length; i++) {
                    const neighborIndex = cellPoints[i];
                    const neighbor = points[neighborIndex];
                    
                    // Check if within radius
                    const distSq = vec2.distanceSq(point, neighbor);
                    if (distSq < this.h * this.h) {
                        neighbors.push(neighborIndex);
                    }
                }
            }
        }
        
        return neighbors;
    }
    
    /**
     * Query neighbors within radius h of a given position
     * @param {Array} position - Query position [x, y]
     * @param {Array} points - Array of all points
     * @returns {Array} Array of neighbor point indices
     */
    queryPosition(position, points) {
        const [gx, gy] = this.getGridCoords(position[0], position[1]);
        
        const neighbors = [];
        
        // Check 3x3 grid of cells around the position
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                const cellIndex = this.getCellIndex(gx + dx, gy + dy);
                const cellPoints = this.cells[cellIndex];
                
                // Check all points in this cell
                for (let i = 0; i < cellPoints.length; i++) {
                    const neighborIndex = cellPoints[i];
                    const neighbor = points[neighborIndex];
                    
                    // Check if within radius
                    const distSq = vec2.distanceSq(position, neighbor);
                    if (distSq < this.h * this.h) {
                        neighbors.push(neighborIndex);
                    }
                }
            }
        }
        
        return neighbors;
    }
    
    /**
     * Get all points in a specific cell
     * @param {number} cellIndex - Cell index
     * @returns {Array} Array of point indices in the cell
     */
    getCellPoints(cellIndex) {
        if (cellIndex >= 0 && cellIndex < this.totalCells) {
            return this.cells[cellIndex];
        }
        return [];
    }
    
    /**
     * Get statistics about the grid
     */
    getStats() {
        let totalPoints = 0;
        let occupiedCells = 0;
        let maxPointsPerCell = 0;
        let minPointsPerCell = Infinity;
        
        for (let i = 0; i < this.totalCells; i++) {
            const cellSize = this.cells[i].length;
            totalPoints += cellSize;
            
            if (cellSize > 0) {
                occupiedCells++;
                maxPointsPerCell = Math.max(maxPointsPerCell, cellSize);
                minPointsPerCell = Math.min(minPointsPerCell, cellSize);
            }
        }
        
        if (occupiedCells === 0) {
            minPointsPerCell = 0;
        }
        
        return {
            totalCells: this.totalCells,
            occupiedCells,
            totalPoints,
            avgPointsPerCell: totalPoints / this.totalCells,
            avgPointsPerOccupiedCell: occupiedCells > 0 ? totalPoints / occupiedCells : 0,
            maxPointsPerCell,
            minPointsPerCell,
            gridDimensions: [this.gridWidth, this.gridHeight],
            cellSize: this.h
        };
    }
    
    /**
     * Debug visualization - get cell boundaries for rendering
     */
    getCellBoundaries() {
        const boundaries = [];
        
        for (let gy = 0; gy <= this.gridHeight; gy++) {
            for (let gx = 0; gx <= this.gridWidth; gx++) {
                const x = gx * this.h;
                const y = gy * this.h;
                
                // Horizontal lines
                if (gy < this.gridHeight) {
                    boundaries.push({
                        type: 'horizontal',
                        x1: x,
                        y1: y,
                        x2: x + this.h,
                        y2: y
                    });
                }
                
                // Vertical lines
                if (gx < this.gridWidth) {
                    boundaries.push({
                        type: 'vertical',
                        x1: x,
                        y1: y,
                        x2: x,
                        y2: y + this.h
                    });
                }
            }
        }
        
        return boundaries;
    }
}
