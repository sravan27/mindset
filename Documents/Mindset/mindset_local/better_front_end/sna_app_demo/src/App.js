import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import axios from 'axios';
import NewsArticlePage from './NewsArticlePage';
import ArticleRead from './ArticleRead';
import ArticleDetail from './ArticleDetail';
import Header from './Header';
import config from './config';
import './App.css';

function App() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch articles when component mounts
    fetchArticles();
  }, []);

  const fetchArticles = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Use a larger limit to ensure we get enough unique articles
      const response = await axios.get(`${config.API_URL}/articles?limit=30`, {
        timeout: config.DEFAULT_TIMEOUT, // Use timeout from config
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      // Check if response contains data
      if (!response.data || !Array.isArray(response.data)) {
        throw new Error('Invalid data format received from server');
      }
      
      // Validate and deduplicate articles
      const articlesData = response.data;
      const uniqueArticles = [];
      const seenIds = new Set();
      const seenTitles = new Set();
      
      articlesData.forEach(article => {
        // Skip if no article_id or title
        if (!article.article_id || !article.title) return;
        
        // Skip duplicate IDs or titles
        if (seenIds.has(article.article_id) || seenTitles.has(article.title)) return;
        
        // Add to tracking sets
        seenIds.add(article.article_id);
        seenTitles.add(article.title);
        
        // Add to unique articles
        uniqueArticles.push(article);
      });
      
      // Check if we have any articles after deduplication
      if (uniqueArticles.length === 0) {
        throw new Error('No valid articles found');
      }
      
      // Update state with unique articles
      setArticles(uniqueArticles);
      setLoading(false);
      
    } catch (err) {
      console.error('Error fetching articles:', err);
      
      // Provide more detailed error messages
      if (err.code === 'ECONNABORTED') {
        setError('Request timed out. The server took too long to respond.');
      } else if (err.response) {
        setError(`Server error (${err.response.status}). Please try again later.`);
      } else if (err.request) {
        setError('No response from server. Please check your internet connection.');
      } else {
        setError(`Failed to fetch articles: ${err.message}`);
      }
      
      setLoading(false);
    }
  };

  return (
    <Router>
      <div className="app-container">
        <Header />
        <Routes>
          <Route path="/" element={
            <NewsArticlePage 
              articles={articles}
              loading={loading}
              error={error}
              refreshArticles={fetchArticles}
            />
          } />
          <Route path="/analyze" element={<ArticleRead />} />
          <Route path="/article/:id" element={<ArticleDetail />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
