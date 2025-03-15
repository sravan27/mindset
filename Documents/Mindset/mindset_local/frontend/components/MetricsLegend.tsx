import React from 'react';

type MetricsLegendProps = {
  compact?: boolean;
};

/**
 * MetricsLegend component explains the meaning of the metrics
 * displayed throughout the application.
 */
const MetricsLegend: React.FC<MetricsLegendProps> = ({ compact = false }) => {
  return (
    <div className={`bg-white rounded-lg shadow p-4 ${compact ? 'text-xs' : 'text-sm'}`}>
      <h3 className={`font-semibold mb-2 ${compact ? 'text-sm' : 'text-base'}`}>
        Understanding MINDSET Metrics
      </h3>
      
      <div className="space-y-3">
        {/* Political Influence */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium">Political Influence</span>
            <div className="flex-grow h-2 rounded bg-gradient-to-r from-political-low to-political-high"></div>
          </div>
          {!compact && (
            <p className="text-gray-600 text-xs">
              Measures political bias from neutral (green) to strong political bias (red).
            </p>
          )}
        </div>
        
        {/* Rhetoric Intensity */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium">Rhetoric Intensity</span>
            <div className="flex-grow h-2 rounded bg-gradient-to-r from-rhetoric-low to-rhetoric-high"></div>
          </div>
          {!compact && (
            <p className="text-gray-600 text-xs">
              Indicates language tone from informative/factual (blue) to emotionally charged (red).
            </p>
          )}
        </div>
        
        {/* Information Depth */}
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium">Information Depth</span>
            <div className="flex items-center gap-1 flex-grow">
              <div className="flex-1 h-2 rounded bg-depth-low"></div>
              <div className="flex-1 h-2 rounded bg-depth-medium"></div>
              <div className="flex-1 h-2 rounded bg-depth-high"></div>
            </div>
          </div>
          {!compact && (
            <div className="text-gray-600 text-xs flex justify-between">
              <span>Overview</span>
              <span>Analysis</span>
              <span>In-depth</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MetricsLegend;