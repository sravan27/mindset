import React from 'react';

const MetricsLegend: React.FC = () => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 mb-6 shadow-sm">
      <h2 className="text-lg font-semibold mb-3">News Transparency Metrics</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Political Influence */}
        <div>
          <div className="flex items-center mb-1">
            <div className="w-4 h-4 bg-red-500 rounded-sm mr-2"></div>
            <h3 className="font-medium">Political Influence Scale (0-10)</h3>
          </div>
          <p className="text-sm text-gray-600 pl-6">
            Measures political bias in content. Higher scores indicate stronger political framing.
          </p>
        </div>
        
        {/* Rhetoric Intensity */}
        <div>
          <div className="flex items-center mb-1">
            <div className="w-4 h-4 bg-yellow-500 rounded-sm mr-2"></div>
            <h3 className="font-medium">Rhetoric Intensity Scale (0-10)</h3>
          </div>
          <p className="text-sm text-gray-600 pl-6">
            Measures emotional and persuasive language. Higher scores indicate more rhetoric and less neutrality.
          </p>
        </div>
        
        {/* Information Depth */}
        <div>
          <div className="flex items-center mb-1">
            <div className="w-4 h-4 bg-green-500 rounded-sm mr-2"></div>
            <h3 className="font-medium">Information Depth Score (0-10)</h3>
          </div>
          <p className="text-sm text-gray-600 pl-6">
            Measures content depth and substance. Higher scores indicate more substantial information content.
          </p>
        </div>
      </div>
      
      <div className="mt-3 text-xs text-gray-500">
        <p>
          Metrics are calculated by the Silicon Layer, combining multiple AI models for transparency and explainability.
        </p>
      </div>
    </div>
  );
};

export default MetricsLegend;