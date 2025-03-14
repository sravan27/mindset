import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import axios from 'axios';
import Link from 'next/link';

// API URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

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
  content?: string;
  category?: string;
  subcategory?: string;
  source?: string;
  published_at?: string;
  metrics: Metrics;
  recommendations?: Recommendation[];
}

interface ExplanationFeature {
  feature: string;
  contribution: number;
  value: number | null;
}

interface MetricExplanation {
  prediction: number;
  base_value: number;
  explanation_text: string[];
  top_features: ExplanationFeature[];
  all_features: ExplanationFeature[];
}

interface ArticleExplanation {
  news_id: string;
  title: string;
  explanations: {
    political_influence?: MetricExplanation;
    rhetoric_intensity?: MetricExplanation;
    information_depth?: MetricExplanation;
  };
}

export default function ArticlePage() {
  const router = useRouter();
  const { id } = router.query;

  const [article, setArticle] = useState<Article | null>(null);
  const [explanation, setExplanation] = useState<ArticleExplanation | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState<boolean>(false);

  // Fetch article details
  useEffect(() => {
    const fetchArticle = async () => {
      if (!id) return;
      
      try {
        setLoading(true);
        const response = await axios.get(`${API_URL}/articles/${id}`);
        setArticle(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching article:', err);
        setError('Failed to load article details. Please try again later.');
        setLoading(false);
      }
    };

    fetchArticle();
  }, [id]);

  // Fetch explanation when requested
  const fetchExplanation = async () => {
    if (!article) return;
    
    try {
      const response = await axios.get(`${API_URL}/explain/${article.news_id}`);
      setExplanation(response.data);
      setShowExplanation(true);
    } catch (err) {
      console.error('Error fetching explanation:', err);
      setError('Failed to load explanation. Please try again later.');
    }
  };

  // Helper function to format date
  const formatDate = (dateString?: string) => {
    if (!dateString) return '';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
      });
    } catch (e) {
      return dateString;
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <Link href="/" className="inline-block mb-6 text-blue-600 hover:text-blue-800">
        ← Back to Articles
      </Link>

      {loading ? (
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-800"></div>
        </div>
      ) : error ? (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
          <p>{error}</p>
        </div>
      ) : article ? (
        <div>
          <article className="bg-white rounded-lg shadow-md overflow-hidden">
            {/* Header */}
            <header className="p-6 bg-gray-50 border-b">
              <h1 className="text-3xl font-bold text-gray-900">{article.title}</h1>
              
              <div className="mt-2 flex flex-wrap text-sm text-gray-500">
                {article.source && (
                  <span className="mr-4">Source: {article.source}</span>
                )}
                {article.published_at && (
                  <span className="mr-4">Published: {formatDate(article.published_at)}</span>
                )}
                {article.category && (
                  <span className="mr-4">Category: {article.category}</span>
                )}
                {article.subcategory && (
                  <span>Subcategory: {article.subcategory}</span>
                )}
              </div>
            </header>

            {/* Content */}
            <div className="p-6">
              {article.abstract && (
                <div className="text-lg font-medium mb-6">{article.abstract}</div>
              )}
              
              {article.content && (
                <div className="prose max-w-none mb-8">{article.content}</div>
              )}
              
              {article.url && (
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800"
                >
                  Read the full article →
                </a>
              )}
            </div>

            {/* Metrics */}
            <div className="p-6 bg-gray-50 border-t border-b">
              <h2 className="text-xl font-bold mb-4">Transparency Metrics</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Political Influence */}
                <div className="bg-white p-4 rounded shadow">
                  <h3 className="font-bold text-red-700 mb-2">Political Influence</h3>
                  <div className="h-4 w-full bg-gray-200 rounded">
                    <div
                      className="h-full bg-red-500 rounded"
                      style={{ width: `${article.metrics.political_influence * 10}%` }}
                    ></div>
                  </div>
                  <p className="mt-2 text-center font-medium">
                    {article.metrics.political_influence.toFixed(1)}/10
                  </p>
                  <p className="mt-1 text-sm text-gray-600">
                    Measures the political bias level in the content
                  </p>
                </div>

                {/* Rhetoric Intensity */}
                <div className="bg-white p-4 rounded shadow">
                  <h3 className="font-bold text-yellow-700 mb-2">Rhetoric Intensity</h3>
                  <div className="h-4 w-full bg-gray-200 rounded">
                    <div
                      className="h-full bg-yellow-500 rounded"
                      style={{ width: `${article.metrics.rhetoric_intensity * 10}%` }}
                    ></div>
                  </div>
                  <p className="mt-2 text-center font-medium">
                    {article.metrics.rhetoric_intensity.toFixed(1)}/10
                  </p>
                  <p className="mt-1 text-sm text-gray-600">
                    Measures emotional and persuasive language
                  </p>
                </div>

                {/* Information Depth */}
                <div className="bg-white p-4 rounded shadow">
                  <h3 className="font-bold text-green-700 mb-2">Information Depth</h3>
                  <div className="h-4 w-full bg-gray-200 rounded">
                    <div
                      className="h-full bg-green-500 rounded"
                      style={{ width: `${article.metrics.information_depth * 10}%` }}
                    ></div>
                  </div>
                  <p className="mt-2 text-center font-medium">
                    {article.metrics.information_depth.toFixed(1)}/10
                  </p>
                  <p className="mt-1 text-sm text-gray-600">
                    Measures content depth and substance
                  </p>
                </div>
              </div>
              
              {/* Explanation Button */}
              <div className="mt-6 text-center">
                <button
                  onClick={fetchExplanation}
                  disabled={showExplanation}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50"
                >
                  {showExplanation ? 'Explanation Loaded' : 'Explain These Metrics'}
                </button>
              </div>
            </div>

            {/* Explanations */}
            {showExplanation && explanation && (
              <div className="p-6 border-b">
                <h2 className="text-xl font-bold mb-4">Metrics Explanation</h2>
                
                <div className="grid grid-cols-1 gap-6">
                  {/* Political Influence Explanation */}
                  {explanation.explanations.political_influence && (
                    <div className="bg-gray-50 p-4 rounded">
                      <h3 className="font-bold text-red-700 mb-2">Political Influence Explanation</h3>
                      
                      {explanation.explanations.political_influence.explanation_text.map((text, idx) => (
                        <p key={idx} className="mb-2">{text}</p>
                      ))}
                      
                      <h4 className="font-medium mt-4 mb-2">Top Contributing Factors:</h4>
                      <ul className="list-disc pl-5">
                        {explanation.explanations.political_influence.top_features.map((feature, idx) => (
                          <li key={idx} className="mb-1">
                            <span className="font-medium">{feature.feature}:</span>{' '}
                            <span className={feature.contribution > 0 ? 'text-red-600' : 'text-blue-600'}>
                              {feature.contribution > 0 ? '+' : ''}{feature.contribution.toFixed(2)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Rhetoric Intensity Explanation */}
                  {explanation.explanations.rhetoric_intensity && (
                    <div className="bg-gray-50 p-4 rounded">
                      <h3 className="font-bold text-yellow-700 mb-2">Rhetoric Intensity Explanation</h3>
                      
                      {explanation.explanations.rhetoric_intensity.explanation_text.map((text, idx) => (
                        <p key={idx} className="mb-2">{text}</p>
                      ))}
                      
                      <h4 className="font-medium mt-4 mb-2">Top Contributing Factors:</h4>
                      <ul className="list-disc pl-5">
                        {explanation.explanations.rhetoric_intensity.top_features.map((feature, idx) => (
                          <li key={idx} className="mb-1">
                            <span className="font-medium">{feature.feature}:</span>{' '}
                            <span className={feature.contribution > 0 ? 'text-red-600' : 'text-blue-600'}>
                              {feature.contribution > 0 ? '+' : ''}{feature.contribution.toFixed(2)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Information Depth Explanation */}
                  {explanation.explanations.information_depth && (
                    <div className="bg-gray-50 p-4 rounded">
                      <h3 className="font-bold text-green-700 mb-2">Information Depth Explanation</h3>
                      
                      {explanation.explanations.information_depth.explanation_text.map((text, idx) => (
                        <p key={idx} className="mb-2">{text}</p>
                      ))}
                      
                      <h4 className="font-medium mt-4 mb-2">Top Contributing Factors:</h4>
                      <ul className="list-disc pl-5">
                        {explanation.explanations.information_depth.top_features.map((feature, idx) => (
                          <li key={idx} className="mb-1">
                            <span className="font-medium">{feature.feature}:</span>{' '}
                            <span className={feature.contribution > 0 ? 'text-green-600' : 'text-red-600'}>
                              {feature.contribution > 0 ? '+' : ''}{feature.contribution.toFixed(2)}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {article.recommendations && article.recommendations.length > 0 && (
              <div className="p-6">
                <h2 className="text-xl font-bold mb-4">Related Articles</h2>
                
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {article.recommendations.map((rec) => (
                    <Link
                      key={rec.news_id}
                      href={`/article/${rec.news_id}`}
                      className="p-4 border rounded hover:bg-gray-50 transition-colors"
                    >
                      <h3 className="font-medium">{rec.title}</h3>
                      <p className="text-sm text-gray-600 mt-1">
                        Relevance: {(rec.score * 100).toFixed(0)}%
                      </p>
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </article>
        </div>
      ) : (
        <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
          <p>Article not found.</p>
        </div>
      )}
    </div>
  );
}