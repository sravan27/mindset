import React from 'react';
import Link from 'next/link';
import { FaChartBar } from 'react-icons/fa';

// Define types for the article and metrics
export type ArticleMetrics = {
  political_influence: number;
  rhetoric_intensity: number;
  information_depth: number;
  information_depth_category: string;
};

export type Article = {
  id: string;
  title: string;
  abstract: string;
  source?: string;
  category?: string;
  published_date?: string;
  image_url?: string;
  metrics: ArticleMetrics;
};

type ArticleCardProps = {
  article: Article;
};

/**
 * ArticleCard component displays a news article card with metrics.
 */
const ArticleCard: React.FC<ArticleCardProps> = ({ article }) => {
  const { id, title, abstract, source, category, published_date, image_url, metrics } = article;
  
  // Format the date
  const formattedDate = published_date 
    ? new Date(published_date).toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
      })
    : '';
  
  // Calculate colors based on metrics
  const getPoliticalColor = (value: number) => {
    // Interpolate between green and red
    const green = Math.round(52 * (1 - value));
    const red = Math.round(239 * value);
    return `rgb(${red}, ${green}, 73)`;
  };
  
  const getRhetoricColor = (value: number) => {
    // Interpolate between blue and red
    const blue = Math.round(59 * (1 - value));
    const red = Math.round(239 * value);
    return `rgb(${red}, 73, ${blue})`;
  };
  
  const getInformationDepthLabel = (category: string) => {
    switch (category.toLowerCase()) {
      case 'overview':
        return <span className="bg-depth-low text-gray-800 text-xs px-2 py-1 rounded">Overview</span>;
      case 'analysis':
        return <span className="bg-depth-medium text-white text-xs px-2 py-1 rounded">Analysis</span>;
      case 'in-depth':
        return <span className="bg-depth-high text-white text-xs px-2 py-1 rounded">In-depth</span>;
      default:
        return null;
    }
  };
  
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300">
      <Link href={`/article/${id}`} className="block">
        {/* Article Image */}
        <div className="h-40 bg-gray-200 relative">
          {image_url ? (
            <img 
              src={image_url} 
              alt={title}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-100">
              <FaChartBar className="text-gray-400 text-4xl" />
            </div>
          )}
          
          {/* Category tag */}
          {category && (
            <span className="absolute top-2 left-2 bg-blue-500 text-white text-xs px-2 py-1 rounded">
              {category}
            </span>
          )}
        </div>
        
        {/* Article Content */}
        <div className="p-4">
          <h3 className="font-bold text-lg mb-2 line-clamp-2">{title}</h3>
          <p className="text-gray-600 text-sm mb-3 line-clamp-3">{abstract}</p>
          
          {/* Source and Date */}
          <div className="flex justify-between items-center text-xs text-gray-500 mb-3">
            {source && <span>{source}</span>}
            {formattedDate && <span>{formattedDate}</span>}
          </div>
          
          {/* Metrics */}
          <div className="space-y-2">
            {/* Political Influence */}
            <div className="flex items-center gap-2">
              <span className="text-xs w-24 text-gray-600">Political</span>
              <div className="flex-grow h-1.5 bg-gray-200 rounded">
                <div 
                  className="h-full rounded" 
                  style={{ 
                    width: `${metrics.political_influence * 100}%`,
                    backgroundColor: getPoliticalColor(metrics.political_influence)
                  }}
                ></div>
              </div>
            </div>
            
            {/* Rhetoric Intensity */}
            <div className="flex items-center gap-2">
              <span className="text-xs w-24 text-gray-600">Rhetoric</span>
              <div className="flex-grow h-1.5 bg-gray-200 rounded">
                <div 
                  className="h-full rounded" 
                  style={{ 
                    width: `${metrics.rhetoric_intensity * 100}%`,
                    backgroundColor: getRhetoricColor(metrics.rhetoric_intensity)
                  }}
                ></div>
              </div>
            </div>
            
            {/* Information Depth */}
            <div className="flex items-center gap-2">
              <span className="text-xs w-24 text-gray-600">Depth</span>
              <div className="flex-grow">
                {getInformationDepthLabel(metrics.information_depth_category)}
              </div>
            </div>
          </div>
        </div>
      </Link>
    </div>
  );
};

export default ArticleCard;