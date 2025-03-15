import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import axios from 'axios';
import ArticleCard, { Article, ArticleMetrics } from '../components/ArticleCard';
import MetricsLegend from '../components/MetricsLegend';
import { FaSearch, FaSpinner, FaChartBar, FaInfoCircle } from 'react-icons/fa';

// Sample articles to show when the app first loads
const sampleArticles: Article[] = [
  {
    id: "sample1",
    title: "Global Leaders Gather for Climate Summit",
    abstract: "World leaders from over 100 countries have gathered in Geneva for the annual climate summit to discuss new policies to combat climate change.",
    source: "World News",
    category: "News",
    published_date: "2023-08-15T14:30:00Z",
    metrics: {
      political_influence: 0.3,
      rhetoric_intensity: 0.2,
      information_depth: 0.7,
      information_depth_category: "Analysis"
    }
  },
  {
    id: "sample2",
    title: "Tech Giants Face Antitrust Investigation",
    abstract: "Major tech companies are under scrutiny as federal regulators launch a comprehensive investigation into potential antitrust violations and market dominance.",
    source: "Tech Today",
    category: "Technology",
    published_date: "2023-08-14T09:15:00Z",
    metrics: {
      political_influence: 0.6,
      rhetoric_intensity: 0.5,
      information_depth: 0.8,
      information_depth_category: "In-depth"
    }
  },
  {
    id: "sample3",
    title: "New Medical Breakthrough Promises Cancer Treatment Revolution",
    abstract: "Researchers announce a groundbreaking discovery that could transform how certain types of cancer are treated, offering hope to millions of patients worldwide.",
    source: "Health News",
    category: "Health",
    published_date: "2023-08-13T16:45:00Z",
    metrics: {
      political_influence: 0.1,
      rhetoric_intensity: 0.4,
      information_depth: 0.9,
      information_depth_category: "In-depth"
    }
  },
  {
    id: "sample4",
    title: "Stock Market Surges to Record Highs",
    abstract: "Wall Street celebrates as major indexes reach unprecedented levels, driven by strong corporate earnings and positive economic indicators.",
    source: "Financial Times",
    category: "Finance",
    published_date: "2023-08-16T10:20:00Z",
    metrics: {
      political_influence: 0.2,
      rhetoric_intensity: 0.3,
      information_depth: 0.5,
      information_depth_category: "Analysis"
    }
  },
  {
    id: "sample5",
    title: "Controversial Voting Bill Sparks Nationwide Protests",
    abstract: "Demonstrators take to the streets in major cities across the country to protest against a controversial voting rights bill that critics claim will restrict access to polls.",
    source: "Politics Daily",
    category: "Politics",
    published_date: "2023-08-12T13:10:00Z",
    metrics: {
      political_influence: 0.9,
      rhetoric_intensity: 0.8,
      information_depth: 0.6,
      information_depth_category: "Analysis"
    }
  },
  {
    id: "sample6",
    title: "Sports Team Wins Championship in Dramatic Finale",
    abstract: "In a thrilling conclusion to the season, underdogs overcome the odds to clinch the championship title in the final seconds of the game.",
    source: "Sports Network",
    category: "Sports",
    published_date: "2023-08-17T22:05:00Z",
    metrics: {
      political_influence: 0.1,
      rhetoric_intensity: 0.7,
      information_depth: 0.3,
      information_depth_category: "Overview"
    }
  }
];

// Define the API URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function HomePage() {
  // State variables
  const [articles, setArticles] = useState<Article[]>(sampleArticles);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [showLegend, setShowLegend] = useState<boolean>(false);
  
  // Function to convert API article to our article format
  const convertApiArticle = (apiArticle: any, metrics: ArticleMetrics): Article => {
    return {
      id: apiArticle.article_id || `article_${Date.now()}`,
      title: apiArticle.title,
      abstract: apiArticle.abstract || '',
      source: apiArticle.source,
      category: apiArticle.category,
      published_date: apiArticle.published_date,
      metrics: metrics
    };
  };
  
  // Function to analyze an article
  const analyzeArticle = async (title: string, abstract: string) => {
    setAnalyzing(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_URL}/analyze`, {
        title,
        abstract,
        article_id: `manual_${Date.now()}`
      });
      
      const metrics = response.data;
      
      // Create a new article with the analyzed metrics
      const newArticle = convertApiArticle(
        { 
          title, 
          abstract, 
          source: "User Input",
          category: "Custom",
          published_date: new Date().toISOString(),
          article_id: metrics.article_id
        }, 
        {
          political_influence: metrics.political_influence,
          rhetoric_intensity: metrics.rhetoric_intensity,
          information_depth: metrics.information_depth,
          information_depth_category: metrics.information_depth_category
        }
      );
      
      // Add the new article to the beginning of the list
      setArticles(prevArticles => [newArticle, ...prevArticles]);
      
      // Clear the search query
      setSearchQuery('');
    } catch (err) {
      console.error('Error analyzing article:', err);
      setError('Failed to analyze article. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };
  
  // Function to handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Split the search query into title and abstract
    // For simplicity, we'll use the first line as title and the rest as abstract
    const lines = searchQuery.trim().split('\n');
    const title = lines[0] || 'Untitled Article';
    const abstract = lines.slice(1).join('\n') || 'No abstract provided.';
    
    analyzeArticle(title, abstract);
  };
  
  // Function to filter articles based on search query
  const filteredArticles = articles.filter(article => 
    article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    article.abstract.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>MINDSET - AI-Powered News Analytics</title>
        <meta name="description" content="Analyze news articles with AI for political influence, rhetoric intensity, and information depth." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center mb-4 md:mb-0">
              <FaChartBar className="text-blue-600 text-3xl mr-2" />
              <h1 className="text-2xl font-bold text-gray-800">MINDSET</h1>
              <span className="ml-2 text-sm text-gray-500">AI-Powered News Analytics</span>
            </div>
            
            <button 
              onClick={() => setShowLegend(!showLegend)} 
              className="flex items-center text-sm text-blue-600 hover:text-blue-800"
            >
              <FaInfoCircle className="mr-1" />
              {showLegend ? 'Hide Metrics Legend' : 'Show Metrics Legend'}
            </button>
          </div>
        </div>
      </header>
      
      <main className="container mx-auto px-4 py-8">
        {/* Metrics Legend */}
        {showLegend && (
          <div className="mb-8">
            <MetricsLegend />
          </div>
        )}
        
        {/* Article Input Form */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Analyze Your Own Content</h2>
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <textarea
                className="w-full p-3 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={5}
                placeholder="Paste or type an article (first line will be used as title, rest as content)..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={analyzing}
              />
            </div>
            <div className="flex justify-end">
              <button
                type="submit"
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors flex items-center disabled:bg-blue-300"
                disabled={analyzing || !searchQuery.trim()}
              >
                {analyzing ? (
                  <>
                    <FaSpinner className="animate-spin mr-2" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <FaSearch className="mr-2" />
                    Analyze
                  </>
                )}
              </button>
            </div>
          </form>
          {error && (
            <div className="mt-4 text-red-500 text-sm">{error}</div>
          )}
        </div>
        
        {/* Articles Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {articles.map((article) => (
            <ArticleCard key={article.id} article={article} />
          ))}
        </div>
        
        {articles.length === 0 && (
          <div className="text-center py-12">
            <FaSearch className="mx-auto text-4xl text-gray-300 mb-4" />
            <h3 className="text-xl font-medium text-gray-500">No articles found</h3>
            <p className="text-gray-400">Try analyzing your own content or check back later for more articles.</p>
          </div>
        )}
      </main>
      
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-gray-500 text-sm">
            MINDSET - AI-Powered News Analytics | &copy; {new Date().getFullYear()}
          </p>
        </div>
      </footer>
    </div>
  );
}