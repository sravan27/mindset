import React from 'react';
import Link from 'next/link';

// Types
interface Metrics {
  political_influence: number;
  rhetoric_intensity: number;
  information_depth: number;
}

interface Recommendation {
  news_id: string;
  title: string;
  score: number;
}

interface Article {
  news_id: string;
  title: string;
  abstract?: string;
  url?: string;
  category?: string;
  subcategory?: string;
  source?: string;
  published_at?: string;
  metrics: Metrics;
  recommendations?: Recommendation[];
}

interface ArticleCardProps {
  article: Article;
}

const ArticleCard: React.FC<ArticleCardProps> = ({ article }) => {
  // Helper function to format date
  const formatDate = (dateString?: string) => {
    if (!dateString) return '';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch (e) {
      return dateString;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200 hover:shadow-lg transition-shadow">
      <div className="p-5">
        {/* Category */}
        {article.category && (
          <div className="text-xs uppercase tracking-wider text-blue-600 mb-2">
            {article.category}
          </div>
        )}
        
        {/* Title */}
        <h2 className="text-xl font-bold mb-2 line-clamp-2">
          <Link href={`/article/${article.news_id}`} className="hover:text-blue-700">
            {article.title}
          </Link>
        </h2>
        
        {/* Abstract */}
        {article.abstract && (
          <p className="text-gray-600 mb-4 line-clamp-3">
            {article.abstract}
          </p>
        )}
        
        {/* Source and Date */}
        <div className="flex items-center text-sm text-gray-500 mb-4">
          {article.source && (
            <span className="mr-3">{article.source}</span>
          )}
          {article.published_at && (
            <span>{formatDate(article.published_at)}</span>
          )}
        </div>
        
        {/* Metrics */}
        <div className="space-y-2 mb-4">
          {/* Political Influence */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium">Political Influence</span>
              <span>{article.metrics.political_influence.toFixed(1)}/10</span>
            </div>
            <div className="h-2 w-full bg-gray-200 rounded overflow-hidden">
              <div
                className="h-full bg-red-500"
                style={{ width: `${article.metrics.political_influence * 10}%` }}
              ></div>
            </div>
          </div>
          
          {/* Rhetoric Intensity */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium">Rhetoric Intensity</span>
              <span>{article.metrics.rhetoric_intensity.toFixed(1)}/10</span>
            </div>
            <div className="h-2 w-full bg-gray-200 rounded overflow-hidden">
              <div
                className="h-full bg-yellow-500"
                style={{ width: `${article.metrics.rhetoric_intensity * 10}%` }}
              ></div>
            </div>
          </div>
          
          {/* Information Depth */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium">Information Depth</span>
              <span>{article.metrics.information_depth.toFixed(1)}/10</span>
            </div>
            <div className="h-2 w-full bg-gray-200 rounded overflow-hidden">
              <div
                className="h-full bg-green-500"
                style={{ width: `${article.metrics.information_depth * 10}%` }}
              ></div>
            </div>
          </div>
        </div>
        
        {/* Read More Link */}
        <Link
          href={`/article/${article.news_id}`}
          className="inline-block text-sm font-medium text-blue-600 hover:text-blue-800"
        >
          Read more →
        </Link>
      </div>
    </div>
  );
};

export default ArticleCard;