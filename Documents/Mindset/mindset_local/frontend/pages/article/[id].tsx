import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import Link from 'next/link';
import axios from 'axios';
import { FaArrowLeft, FaChartBar, FaExclamationTriangle, FaInfoCircle } from 'react-icons/fa';
import { Article, ArticleMetrics } from '../../components/ArticleCard';
import MetricsLegend from '../../components/MetricsLegend';

// Define the API URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Sample articles to show when the app first loads (same as on index page)
const sampleArticles: Article[] = [
  {
    id: "sample1",
    title: "Global Leaders Gather for Climate Summit",
    abstract: "World leaders from over 100 countries have gathered in Geneva for the annual climate summit to discuss new policies to combat climate change.",
    source: "World News",
    category: "News",
    published_date: "2023-08-15T14:30:00Z",
    content: "GENEVA — World leaders from over 100 countries have gathered in Geneva for the annual climate summit to discuss new policies to combat climate change. The summit, which runs for three days, will focus on reducing carbon emissions and transitioning to renewable energy sources.\n\nThe conference comes amid growing concern about extreme weather events and rising sea levels attributed to climate change. Scientists warn that immediate action is needed to prevent catastrophic environmental consequences.\n\n\"This summit represents a crucial opportunity for global cooperation,\" said UN Secretary-General António Guterres in his opening remarks. \"The time for incremental steps has passed. We need bold, transformative action.\"\n\nLeaders from major carbon-emitting countries are expected to announce new commitments to reduce greenhouse gas emissions. Environmental activists are also present, calling for more ambitious targets and concrete plans.\n\nThe summit will conclude with a joint declaration outlining agreed-upon goals and initiatives to address the climate crisis over the next decade.",
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
    content: "WASHINGTON — Major tech companies are under intense scrutiny as federal regulators have launched a comprehensive investigation into potential antitrust violations and market dominance. The investigation, announced yesterday, will examine whether the largest technology firms have engaged in anti-competitive practices that harm consumers and stifle innovation.\n\nThe Department of Justice and Federal Trade Commission will divide responsibility for investigating specific companies, with particular focus on their acquisition histories, data collection practices, and treatment of competing services on their platforms.\n\n\"We are looking at whether these companies have achieved their dominant positions through anti-competitive means,\" said FTC Chairperson Lina Khan. \"Our goal is to ensure that the digital marketplace remains fair and competitive.\"\n\nTech industry representatives have defended their business practices, arguing that they have created products that consumers love while continuously innovating. \"Competition in the tech sector has never been more intense,\" said the CEO of one major platform. \"Consumers have more choices than ever before.\"\n\nThe investigation is expected to last at least a year and could potentially lead to enforcement actions ranging from business practice changes to structural remedies like company breakups. Similar investigations are ongoing in the European Union and United Kingdom.\n\nThis represents the most significant regulatory challenge the tech industry has faced in decades, and analysts suggest it could reshape the digital landscape for years to come.",
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
    content: "BOSTON — In what medical experts are calling a potential paradigm shift in cancer treatment, researchers at the Dana-Farber Cancer Institute have announced a groundbreaking discovery that could transform how certain types of cancer are treated, offering new hope to millions of patients worldwide.\n\nThe research team, led by Dr. Sarah Chen, has developed a novel immunotherapy approach that effectively teaches the body's immune system to identify and destroy cancer cells with unprecedented precision. Early clinical trials have shown remarkable results, with complete remission observed in 72% of patients with previously treatment-resistant forms of lymphoma.\n\n\"What makes this approach revolutionary is its ability to adapt to cancer's notorious tendency to mutate,\" explained Dr. Chen. \"The therapy essentially evolves alongside the cancer, continuing to recognize and eliminate malignant cells even as they attempt to change their molecular signatures to evade detection.\"\n\nThe treatment involves extracting immune cells from patients, genetically modifying them to better recognize cancer-specific biomarkers, and then reintroducing them into the patient's body. Unlike previous immunotherapies, this approach incorporates machine learning algorithms to predict how cancer cells might evolve, programming the immune cells to recognize not just existing cancer patterns but potential future variations as well.\n\nClinical trials are now being expanded to include patients with other difficult-to-treat cancers, including pancreatic and certain types of brain tumors. Regulatory agencies have already designated the treatment for accelerated review given its promising results.\n\n\"While we must be cautious about declaring victory against such a formidable disease, these results represent the most significant advancement in cancer immunotherapy in at least a decade,\" said Dr. Robert Alverez, director of the National Cancer Institute, who was not involved in the research.\n\nThe team estimates that, pending regulatory approval, the treatment could become widely available within three to five years. Research is also underway to make the procedure more affordable and accessible to patients worldwide.",
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
    content: "NEW YORK — Wall Street celebrated today as major stock indexes surged to record highs, driven by stronger-than-expected corporate earnings reports and a series of positive economic indicators. The S&P 500 closed up 2.3%, while the Nasdaq Composite gained 2.7%, both setting new all-time highs.\n\nTech and financial sectors led the rally, with several major companies reporting quarterly results that exceeded analyst expectations. Investor confidence was further boosted by new economic data showing robust consumer spending and moderating inflation.\n\n\"We're seeing a perfect storm of positive factors,\" said Marcus Williams, chief market strategist at Capital Investments. \"Corporate America is demonstrating remarkable resilience and adaptability, and the economic foundation appears increasingly stable.\"\n\nThe strong market performance comes despite ongoing concerns about global supply chain disruptions and labor shortages in some sectors. Analysts suggest that companies have largely adapted to these challenges, implementing technological solutions and adjusting business models accordingly.\n\nThe Federal Reserve's recent comments indicating a measured approach to interest rate adjustments has also contributed to market optimism. Treasury yields remained relatively stable throughout the trading session.\n\nWhile market sentiment is overwhelmingly positive, some analysts urge caution. \"This level of growth isn't indefinitely sustainable,\" noted financial commentator Rebecca Chen. \"Investors should remain vigilant about potential volatility, particularly as we monitor how the economy absorbs the impact of previous policy adjustments.\"",
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
    content: "WASHINGTON — Tens of thousands of demonstrators took to the streets in major cities across the country over the weekend to protest against the controversial Federal Election Security Act, which critics claim will significantly restrict access to polls for millions of voters. The demonstrations mark the largest nationwide protest movement since 2020.\n\nThe bill, which passed along party lines in the House of Representatives last week and now awaits Senate consideration, includes provisions requiring specific types of photo identification, limiting early voting periods, and implementing stricter rules for mail-in ballots. Supporters argue these measures are necessary to ensure election integrity and prevent fraud.\n\n\"This legislation is a direct assault on our democracy,\" said civil rights attorney Marcus Johnson, addressing protesters in Atlanta. \"It creates unnecessary barriers that will disproportionately affect communities of color, the elderly, and low-income voters.\"\n\nProponents of the bill reject these characterizations. \"This is about protecting the sanctity of every legitimate vote,\" said Representative Thomas Williams, one of the bill's sponsors. \"The provisions are reasonable safeguards that most Americans support.\"\n\nThe protests remained largely peaceful, though authorities reported isolated incidents of confrontation between demonstrators and counter-protesters in several cities. Organizers have announced plans for continued demonstrations and a major march on Washington should the bill advance in the Senate.\n\nPolitical analysts suggest the controversy could significantly impact the upcoming midterm elections, mobilizing voters on both sides of the issue. Several legal challenges to the legislation are already being prepared, with opponents arguing that key provisions violate constitutional protections.\n\nA recent national poll showed the country deeply divided on the issue, with 48% supporting the legislation and 46% opposing it, with the remainder undecided.",
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
    content: "MIAMI — In a thrilling conclusion to an unforgettable season, the underdog Portland Pioneers overcame seemingly impossible odds to clinch the championship title in the final seconds of tonight's game against the favored Miami Marlins.\n\nTrailing by four points with just 12 seconds remaining on the clock, Pioneers point guard Jason Reynolds executed a perfect three-point shot under immense pressure. Then, in a moment that will surely be replayed for years to come, team captain Marcus Dawson intercepted a pass and scored the winning basket as the final buzzer sounded, sending the arena into absolute pandemonium.\n\n\"I still can't believe what just happened,\" said a visibly emotional Dawson in the post-game interview. \"We never stopped believing, even when everyone else counted us out.\"\n\nThe victory caps an improbable playoff run for the Pioneers, who entered the tournament as the lowest-seeded team. Head coach Sarah Williams, in just her second season with the team, has now secured her place in the sport's history books.\n\n\"This team has shown extraordinary heart and resilience all season,\" Williams said. \"Tonight was the perfect ending to a journey that's been all about defying expectations.\"\n\nFans poured into downtown Portland to celebrate the city's first major sports championship in 27 years. The team is scheduled to return tomorrow, with a victory parade planned for this weekend.",
    metrics: {
      political_influence: 0.1,
      rhetoric_intensity: 0.7,
      information_depth: 0.3,
      information_depth_category: "Overview"
    }
  }
];

// Explanation interface
interface ExplanationData {
  base_value: number;
  prediction: number;
  contributions: Record<string, number>;
  top_positive: Record<string, number>;
  top_negative: Record<string, number>;
}

// Explanation component
const MetricExplanation: React.FC<{
  title: string;
  explanation: ExplanationData | undefined;
  isPending: boolean;
  error: string | null;
}> = ({ title, explanation, isPending, error }) => {
  if (isPending) {
    return (
      <div className="p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium mb-2">{title} Explanation</h3>
        <p className="text-gray-500 text-sm">Loading explanation...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="p-4 bg-red-50 rounded-lg">
        <h3 className="font-medium mb-2 flex items-center">
          <FaExclamationTriangle className="text-red-500 mr-2" />
          {title} Explanation Error
        </h3>
        <p className="text-red-500 text-sm">{error}</p>
      </div>
    );
  }
  
  if (!explanation) {
    return (
      <div className="p-4 bg-gray-50 rounded-lg">
        <h3 className="font-medium mb-2">{title} Explanation</h3>
        <p className="text-gray-500 text-sm">No explanation available.</p>
      </div>
    );
  }
  
  return (
    <div className="p-4 bg-gray-50 rounded-lg">
      <h3 className="font-medium mb-3">{title} Explanation</h3>
      
      <div className="mb-4">
        <div className="text-sm font-medium mb-1">Top factors increasing {title.toLowerCase()}:</div>
        <div className="space-y-1">
          {Object.entries(explanation.top_positive).length > 0 ? (
            Object.entries(explanation.top_positive).map(([feature, value]) => (
              <div key={feature} className="flex justify-between text-xs">
                <span className="text-gray-700">{feature}</span>
                <span className="text-green-600">+{value.toFixed(2)}</span>
              </div>
            ))
          ) : (
            <p className="text-gray-500 text-xs">No significant positive factors</p>
          )}
        </div>
      </div>
      
      <div>
        <div className="text-sm font-medium mb-1">Top factors decreasing {title.toLowerCase()}:</div>
        <div className="space-y-1">
          {Object.entries(explanation.top_negative).length > 0 ? (
            Object.entries(explanation.top_negative).map(([feature, value]) => (
              <div key={feature} className="flex justify-between text-xs">
                <span className="text-gray-700">{feature}</span>
                <span className="text-red-600">{value.toFixed(2)}</span>
              </div>
            ))
          ) : (
            <p className="text-gray-500 text-xs">No significant negative factors</p>
          )}
        </div>
      </div>
      
      <div className="mt-3 pt-3 border-t border-gray-200">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Base value: {explanation.base_value.toFixed(2)}</span>
          <span>Final score: {explanation.prediction.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
};

export default function ArticleDetailPage() {
  const router = useRouter();
  const { id } = router.query;
  
  // State variables
  const [article, setArticle] = useState<Article | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showLegend, setShowLegend] = useState<boolean>(false);
  
  // Explanation states
  const [explanation, setExplanation] = useState<any>(null);
  const [loadingExplanation, setLoadingExplanation] = useState<boolean>(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);
  
  // Get article by ID from sample data or API
  useEffect(() => {
    if (!id) return;
    
    setLoading(true);
    setError(null);
    
    // For sample articles, just get from the array
    const sampleArticle = sampleArticles.find(a => a.id === id);
    if (sampleArticle) {
      setArticle(sampleArticle);
      setLoading(false);
      return;
    }
    
    // For real articles, we would fetch from the API
    // This is just a placeholder; in a real app you would fetch from your backend
    setLoading(false);
    setError("Article not found");
    
  }, [id]);
  
  // Function to load explanation from API
  const loadExplanation = async () => {
    if (!article) return;
    
    setLoadingExplanation(true);
    setExplanationError(null);
    
    try {
      // For sample articles, just generate a mock explanation
      if (article.id.startsWith('sample')) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Create a mock explanation
        const mockExplanation = {
          political_influence: {
            base_value: 0.5,
            prediction: article.metrics.political_influence,
            contributions: {
              "title_length": 0.05,
              "word_count": 0.1,
              "political_terms": article.metrics.political_influence - 0.3,
              "source_bias": 0.15,
              "named_entities": -0.05
            },
            top_positive: {
              "political_terms": article.metrics.political_influence - 0.3,
              "source_bias": 0.15,
              "title_length": 0.05
            },
            top_negative: {
              "named_entities": -0.05
            }
          },
          rhetoric_intensity: {
            base_value: 0.4,
            prediction: article.metrics.rhetoric_intensity,
            contributions: {
              "emotional_words": article.metrics.rhetoric_intensity - 0.25,
              "exclamation_marks": 0.05,
              "adjective_count": 0.1,
              "sentence_structure": 0.08,
              "sentiment_score": -0.03
            },
            top_positive: {
              "emotional_words": article.metrics.rhetoric_intensity - 0.25,
              "adjective_count": 0.1,
              "sentence_structure": 0.08
            },
            top_negative: {
              "sentiment_score": -0.03
            }
          },
          information_depth: {
            base_value: 0.5,
            prediction: article.metrics.information_depth,
            contributions: {
              "word_count": article.metrics.information_depth - 0.3,
              "unique_words_ratio": 0.15,
              "source_credibility": 0.1,
              "citation_count": 0.08,
              "technical_terms": -0.05
            },
            top_positive: {
              "word_count": article.metrics.information_depth - 0.3,
              "unique_words_ratio": 0.15,
              "source_credibility": 0.1
            },
            top_negative: {
              "technical_terms": -0.05
            }
          }
        };
        
        setExplanation(mockExplanation);
      } else {
        // For real articles, fetch from API
        const response = await axios.post(`${API_URL}/explain`, {
          article_id: article.id,
          title: article.title,
          abstract: article.abstract,
          content: article.content
        });
        
        setExplanation(response.data.explanation);
      }
    } catch (err) {
      console.error('Error loading explanation:', err);
      setExplanationError('Failed to load explanation. Please try again.');
    } finally {
      setLoadingExplanation(false);
    }
  };
  
  // Format the date
  const formattedDate = article?.published_date 
    ? new Date(article.published_date).toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'long', 
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
    switch (category?.toLowerCase()) {
      case 'overview':
        return <span className="bg-depth-low text-gray-800 px-2 py-1 rounded">Overview</span>;
      case 'analysis':
        return <span className="bg-depth-medium text-white px-2 py-1 rounded">Analysis</span>;
      case 'in-depth':
        return <span className="bg-depth-high text-white px-2 py-1 rounded">In-depth</span>;
      default:
        return null;
    }
  };
  
  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>{article ? article.title : 'Article'} | MINDSET</title>
        <meta name="description" content={article?.abstract || 'Article details'} />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center mb-4 md:mb-0">
              <Link href="/" className="flex items-center text-blue-600 hover:text-blue-800">
                <FaArrowLeft className="mr-2" />
                <span className="font-medium">Back to Articles</span>
              </Link>
            </div>
            
            <div className="flex items-center">
              <FaChartBar className="text-blue-600 text-2xl mr-2" />
              <h1 className="text-xl font-bold text-gray-800">MINDSET</h1>
              <span className="ml-2 text-sm text-gray-500">AI-Powered News Analytics</span>
            </div>
            
            <button 
              onClick={() => setShowLegend(!showLegend)} 
              className="flex items-center text-sm text-blue-600 hover:text-blue-800 mt-4 md:mt-0"
            >
              <FaInfoCircle className="mr-1" />
              {showLegend ? 'Hide Metrics Legend' : 'Show Metrics Legend'}
            </button>
          </div>
        </div>
      </header>
      
      <main className="container mx-auto px-4 py-8">
        {loading ? (
          <div className="text-center py-12">
            <p className="text-gray-500">Loading article...</p>
          </div>
        ) : error ? (
          <div className="text-center py-12">
            <FaExclamationTriangle className="mx-auto text-4xl text-red-300 mb-4" />
            <h3 className="text-xl font-medium text-gray-500">{error}</h3>
            <Link href="/" className="text-blue-600 hover:text-blue-800 mt-4 inline-block">
              Return to articles
            </Link>
          </div>
        ) : article ? (
          <div className="flex flex-col lg:flex-row gap-8">
            {/* Article Content */}
            <div className="w-full lg:w-2/3">
              {/* Article Header */}
              <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6">
                {article.image_url && (
                  <img 
                    src={article.image_url} 
                    alt={article.title}
                    className="w-full h-64 object-cover"
                  />
                )}
                
                <div className="p-6">
                  {article.category && (
                    <span className="inline-block bg-blue-500 text-white text-xs px-2 py-1 rounded mb-4">
                      {article.category}
                    </span>
                  )}
                  
                  <h1 className="text-3xl font-bold text-gray-800 mb-4">{article.title}</h1>
                  
                  <div className="flex flex-col md:flex-row md:items-center gap-2 md:gap-4 text-sm text-gray-500 mb-6">
                    {article.source && <span>{article.source}</span>}
                    {formattedDate && <span>{formattedDate}</span>}
                  </div>
                  
                  <div className="text-lg text-gray-600 mb-4 italic">
                    {article.abstract}
                  </div>
                </div>
              </div>
              
              {/* Article Body */}
              <div className="bg-white rounded-lg shadow-md p-6 mb-6">
                <div className="prose max-w-none">
                  {article.content ? (
                    article.content.split('\n\n').map((paragraph, index) => (
                      <p key={index} className="mb-4">{paragraph}</p>
                    ))
                  ) : (
                    <p className="text-gray-500">No content available for this article.</p>
                  )}
                </div>
              </div>
            </div>
            
            {/* Metrics Sidebar */}
            <div className="w-full lg:w-1/3 space-y-6">
              {/* Metrics Legend */}
              {showLegend && (
                <MetricsLegend />
              )}
              
              {/* Article Metrics */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold mb-4">Article Metrics</h2>
                
                <div className="space-y-6">
                  {/* Political Influence */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-medium">Political Influence</h3>
                      <span className="text-sm">
                        {(article.metrics.political_influence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-3 bg-gray-200 rounded">
                      <div 
                        className="h-full rounded" 
                        style={{ 
                          width: `${article.metrics.political_influence * 100}%`,
                          backgroundColor: getPoliticalColor(article.metrics.political_influence)
                        }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Neutral</span>
                      <span>Strong political bias</span>
                    </div>
                  </div>
                  
                  {/* Rhetoric Intensity */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-medium">Rhetoric Intensity</h3>
                      <span className="text-sm">
                        {(article.metrics.rhetoric_intensity * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-3 bg-gray-200 rounded">
                      <div 
                        className="h-full rounded" 
                        style={{ 
                          width: `${article.metrics.rhetoric_intensity * 100}%`,
                          backgroundColor: getRhetoricColor(article.metrics.rhetoric_intensity)
                        }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Informative</span>
                      <span>Emotionally charged</span>
                    </div>
                  </div>
                  
                  {/* Information Depth */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-medium">Information Depth</h3>
                      <span>
                        {getInformationDepthLabel(article.metrics.information_depth_category)}
                      </span>
                    </div>
                    <div className="flex gap-1 h-3">
                      <div 
                        className={`flex-1 rounded ${
                          article.metrics.information_depth_category.toLowerCase() === 'overview' 
                            ? 'bg-depth-low' 
                            : 'bg-gray-200'
                        }`}
                      ></div>
                      <div 
                        className={`flex-1 rounded ${
                          article.metrics.information_depth_category.toLowerCase() === 'analysis' 
                            ? 'bg-depth-medium' 
                            : 'bg-gray-200'
                        }`}
                      ></div>
                      <div 
                        className={`flex-1 rounded ${
                          article.metrics.information_depth_category.toLowerCase() === 'in-depth' 
                            ? 'bg-depth-high' 
                            : 'bg-gray-200'
                        }`}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>Overview</span>
                      <span>Analysis</span>
                      <span>In-depth</span>
                    </div>
                  </div>
                </div>
                
                {/* Explanation Button */}
                {!explanation && !loadingExplanation && (
                  <button
                    onClick={loadExplanation}
                    className="mt-6 w-full bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition-colors flex items-center justify-center"
                  >
                    <FaInfoCircle className="mr-2" />
                    Explain These Metrics
                  </button>
                )}
              </div>
              
              {/* Explanations */}
              {(explanation || loadingExplanation || explanationError) && (
                <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
                  <h2 className="text-xl font-semibold mb-4">Metric Explanations</h2>
                  
                  <MetricExplanation
                    title="Political Influence"
                    explanation={explanation?.political_influence}
                    isPending={loadingExplanation}
                    error={explanationError}
                  />
                  
                  <MetricExplanation
                    title="Rhetoric Intensity"
                    explanation={explanation?.rhetoric_intensity}
                    isPending={loadingExplanation}
                    error={explanationError}
                  />
                  
                  <MetricExplanation
                    title="Information Depth"
                    explanation={explanation?.information_depth}
                    isPending={loadingExplanation}
                    error={explanationError}
                  />
                </div>
              )}
            </div>
          </div>
        ) : null}
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