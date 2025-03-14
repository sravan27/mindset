import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ArticleCard from '../components/ArticleCard';
import MetricsLegend from '../components/MetricsLegend';

// API URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Article type
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

// Metrics summary type
interface MetricsSummary {
  count: number;
  averages: Metrics;
}

export default function Home() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [metricsSummary, setMetricsSummary] = useState<MetricsSummary | null>(null);
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('');

  // Fetch articles from API
  useEffect(() => {
    const fetchArticles = async () => {
      try {
        // Add category filter if selected
        let url = `${API_URL}/articles?limit=20`;
        if (selectedCategory) {
          url += `&category=${selectedCategory}`;
        }

        const response = await axios.get(url);
        setArticles(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching articles:', err);
        setError('Failed to load articles. Please try again later.');
        setLoading(false);
      }
    };

    fetchArticles();
  }, [selectedCategory]);

  // Fetch metrics summary
  useEffect(() => {
    const fetchMetricsSummary = async () => {
      try {
        const response = await axios.get(`${API_URL}/metrics`);
        setMetricsSummary(response.data);
      } catch (err) {
        console.error('Error fetching metrics summary:', err);
      }
    };

    fetchMetricsSummary();
  }, []);

  // Fetch categories
  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const response = await axios.get(`${API_URL}/categories`);
        setCategories(response.data.categories);
      } catch (err) {
        console.error('Error fetching categories:', err);
      }
    };

    fetchCategories();
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <header className="mb-8">
        <h1 className="text-4xl font-bold text-blue-800 mb-2">MINDSET News Analytics</h1>
        <p className="text-xl text-gray-600 mb-6">
          Transparent and explainable news analysis platform
        </p>
        
        {/* Metrics Legend */}
        <MetricsLegend />
        
        {/* Category Filter */}
        {categories.length > 0 && (
          <div className="my-4">
            <label htmlFor="category-filter" className="mr-2 font-medium">
              Filter by category:
            </label>
            <select
              id="category-filter"
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="border p-2 rounded"
            >
              <option value="">All Categories</option>
              {categories.map((category) => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>
          </div>
        )}
        
        {/* Metrics Summary */}
        {metricsSummary && (
          <div className="bg-gray-100 p-4 rounded-lg my-4">
            <h2 className="text-lg font-semibold mb-2">Average Metrics ({metricsSummary.count} articles)</h2>
            <div className="flex flex-wrap justify-between">
              <div className="w-full md:w-1/3 p-2">
                <div className="bg-white p-3 rounded shadow-sm">
                  <h3 className="font-medium">Political Influence</h3>
                  <div className="h-4 w-full bg-gray-200 rounded mt-1">
                    <div
                      className="h-full bg-red-500 rounded"
                      style={{ width: `${metricsSummary.averages.political_influence * 10}%` }}
                    ></div>
                  </div>
                  <p className="text-sm mt-1">{metricsSummary.averages.political_influence.toFixed(1)}/10</p>
                </div>
              </div>
              <div className="w-full md:w-1/3 p-2">
                <div className="bg-white p-3 rounded shadow-sm">
                  <h3 className="font-medium">Rhetoric Intensity</h3>
                  <div className="h-4 w-full bg-gray-200 rounded mt-1">
                    <div
                      className="h-full bg-yellow-500 rounded"
                      style={{ width: `${metricsSummary.averages.rhetoric_intensity * 10}%` }}
                    ></div>
                  </div>
                  <p className="text-sm mt-1">{metricsSummary.averages.rhetoric_intensity.toFixed(1)}/10</p>
                </div>
              </div>
              <div className="w-full md:w-1/3 p-2">
                <div className="bg-white p-3 rounded shadow-sm">
                  <h3 className="font-medium">Information Depth</h3>
                  <div className="h-4 w-full bg-gray-200 rounded mt-1">
                    <div
                      className="h-full bg-green-500 rounded"
                      style={{ width: `${metricsSummary.averages.information_depth * 10}%` }}
                    ></div>
                  </div>
                  <p className="text-sm mt-1">{metricsSummary.averages.information_depth.toFixed(1)}/10</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </header>

      <main>
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-800"></div>
          </div>
        ) : error ? (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <p>{error}</p>
          </div>
        ) : articles.length === 0 ? (
          <div className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
            <p>No articles available. Please check the API connection or try again later.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {articles.map((article) => (
              <ArticleCard key={article.news_id} article={article} />
            ))}
          </div>
        )}
      </main>

      <footer className="mt-16 pt-8 border-t border-gray-300 text-center text-gray-600">
        <p>&copy; {new Date().getFullYear()} MINDSET News Analytics</p>
        <p className="text-sm mt-2">
          Transparent news analysis with Silicon Layer technology
        </p>
      </footer>
    </div>
  );
}