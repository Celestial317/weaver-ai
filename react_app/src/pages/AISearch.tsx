import React, { useState } from 'react';
import { Search, Heart, ShoppingBag } from 'lucide-react';

interface SearchResult {
  id: number;
  image_id: string;
  name: string;
  category: string;
  image: string;
  combination_type: string;
  matching_filters: Record<string, string>;
  reason: string;
  search_order?: number;
}

interface SearchResponse {
  recommendations: SearchResult[];
  total_found: number;
  total_available?: number;
  strategy: string;
  combinations_tried: string[];
  extracted_attributes: Record<string, string>;
  search_coverage: string;
  is_progressive?: boolean;
  is_expanded?: boolean;
  remaining_combinations?: number;
  has_more?: boolean;
  next_offset?: number;
  feature_breakdown?: {
    "4_features": number;
    "3_features": number;
    "2_features": number;
    "1_feature": number;
  };
  error: string | null;
}

const AISearch: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchInfo, setSearchInfo] = useState<SearchResponse | null>(null);
  const [searchMode, setSearchMode] = useState<'progressive' | 'complete' | 'continuous'>('continuous');
  const [canExpandSearch, setCanExpandSearch] = useState(false);
  const [expandedResults, setExpandedResults] = useState<SearchResult[]>([]);
  const [isExpanding, setIsExpanding] = useState(false);
  const [currentOffset, setCurrentOffset] = useState(0);
  const [allResults, setAllResults] = useState<SearchResult[]>([]);
  const [hasMoreResults, setHasMoreResults] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setResults([]);
    setSearchInfo(null);
    setExpandedResults([]);
    setCanExpandSearch(false);
    setAllResults([]);
    setCurrentOffset(0);
    setHasMoreResults(false);

    try {
      let endpoint = '/search';
      if (searchMode === 'progressive') endpoint = '/search/progressive';
      else if (searchMode === 'complete') endpoint = '/search';
      else if (searchMode === 'continuous') endpoint = '/search/continuous';

      const response = await fetch(`http://localhost:8002${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          limit: 8,
          offset: 0
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SearchResponse = await response.json();
      
      if (searchMode === 'continuous') {
        setAllResults(data.recommendations);
        setCurrentOffset(data.next_offset || data.recommendations.length);
        setHasMoreResults(data.has_more || false);
        setResults(data.recommendations);
      } else {
        setResults(data.recommendations);
        // Enable "Search More" if there are results and it's not an expanded search
        if (data.recommendations.length > 0 && !data.is_expanded) {
          setCanExpandSearch(true);
        }
      }

      setSearchInfo(data);

    } catch (error) {
      console.error('Search error:', error);
      setSearchInfo({
        recommendations: [],
        total_found: 0,
        strategy: 'Search failed',
        combinations_tried: [],
        extracted_attributes: {},
        search_coverage: '0 of 0 combinations',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleShowMore = async () => {
    if (!query.trim() || !hasMoreResults) return;

    setIsExpanding(true);

    try {
      const response = await fetch('http://localhost:8002/search/continuous', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          limit: 8,  // Load 8 more items
          offset: currentOffset
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SearchResponse = await response.json();
      
      // Add new results to existing ones
      const newResults = data.recommendations;
      setAllResults(prev => [...prev, ...newResults]);
      setCurrentOffset(data.next_offset || currentOffset + newResults.length);
      setHasMoreResults(data.has_more || false);
      
      // Update search info to show more details
      if (data.strategy) {
        setSearchInfo(prev => prev ? {
          ...prev,
          strategy: data.strategy,
          total_available: data.total_available,
          feature_breakdown: data.feature_breakdown
        } : data);
      }

    } catch (error) {
      console.error('Show More error:', error);
    } finally {
      setIsExpanding(false);
    }
  };

  const handleSearchMore = async () => {
    if (!query.trim()) return;

    setIsExpanding(true);

    try {
      const response = await fetch('http://localhost:8002/search/more', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          limit: 12  // Get more results for expansion
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SearchResponse = await response.json();
      
      // Add new expanded results, avoiding duplicates
      const newResults = data.recommendations.filter(
        newItem => !results.some(existingItem => existingItem.id === newItem.id) &&
                   !expandedResults.some(existingItem => existingItem.id === newItem.id)
      );
      
      setExpandedResults(prev => [...prev, ...newResults]);
      
      // Update search info to show expansion details
      if (data.strategy) {
        setSearchInfo(prev => prev ? {
          ...prev,
          strategy: `${prev.strategy} + ${data.strategy}`,
          combinations_tried: [...prev.combinations_tried, ...data.combinations_tried],
          search_coverage: `${prev.search_coverage} + ${data.search_coverage}`
        } : data);
      }

      // Disable further expansion after expanding
      setCanExpandSearch(false);

    } catch (error) {
      console.error('Search More error:', error);
    } finally {
      setIsExpanding(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const exampleQueries = [
    "full sleeve Floral dress blue colour",
    "red shirt with buttons",
    "casual black jeans",
    "formal blazer navy blue",
    "striped t-shirt short sleeve"
  ];

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <h1 className="text-4xl lg:text-5xl font-bold text-white">Search Engine</h1>
          </div>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Describe what you're looking in natural language, AI will find the perfect clothing matches using intelligent search combinations.
          </p>
        </div>

        {/* Search Mode Toggle */}
        <div className="flex justify-center mb-6">
          <div className="bg-gray-800 rounded-lg p-1 flex">
            <button
              onClick={() => setSearchMode('continuous')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                searchMode === 'continuous'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              Continuous
            </button>
            <button
              onClick={() => setSearchMode('progressive')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                searchMode === 'progressive'
                  ? 'bg-orange-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              Quick Results
            </button>
            <button
              onClick={() => setSearchMode('complete')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                searchMode === 'complete'
                  ? 'bg-orange-600 text-white'
                  : 'text-gray-300 hover:text-white'
              }`}
            >
              All Combinations
            </button>
          </div>
        </div>

        {/* Search Section */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 mb-8">
          <div className="flex items-center mb-6">
            <Search className="w-6 h-6 text-orange-500 mr-3" />
            <h2 className="text-2xl font-semibold text-white">Search Fashion Items</h2>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Describe what you're looking for
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="e.g., 'full sleeve floral dress blue colour', 'red shirt with buttons', 'casual black jeans'..."
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none"
                rows={3}
              />
            </div>
            
            <button
              onClick={handleSearch}
              disabled={!query.trim() || loading}
              className="w-full py-3 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  <span className="ml-2">Searching with AI...</span>
                </>
              ) : (
                <>
                  <Search className="w-5 h-5 mr-2" />
                  Search Fashion Items
                </>
              )}
            </button>
          </div>

          {/* Search Examples */}
          <div className="mt-6 p-4 bg-gray-700/30 rounded-lg">
            <h4 className="text-sm font-semibold text-white mb-2">Search Examples:</h4>
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((example, index) => (
                <button
                  key={index}
                  onClick={() => setQuery(example)}
                  className="px-3 py-1 bg-gray-700 text-gray-300 rounded-full text-sm hover:bg-gray-600 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Search Info */}
        {searchInfo && (
          <div className="mb-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
            <div className="flex items-center mb-2">
              <h3 className="text-lg font-semibold text-white">AI Search Strategy</h3>
            </div>
            <p className="text-gray-300 text-sm mb-2">{searchInfo.strategy}</p>
            <p className="text-gray-400 text-xs">Coverage: {searchInfo.search_coverage}</p>
            
            {/* Feature Breakdown for Continuous Mode */}
            {searchMode === 'continuous' && searchInfo.feature_breakdown && (
              <div className="mt-3 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                <p className="text-purple-300 text-xs mb-2 font-semibold">Results by Feature Count:</p>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  {Object.entries(searchInfo.feature_breakdown)
                    .sort(([a], [b]) => {
                      const numA = parseInt(a.split('_')[0]);
                      const numB = parseInt(b.split('_')[0]);
                      return numB - numA; // Sort descending (highest features first)
                    })
                    .map(([key, count], index) => {
                      const featureNum = key.split('_')[0];
                      const colors = ['bg-purple-600', 'bg-pink-600', 'bg-blue-600', 'bg-green-500', 'bg-yellow-500', 'bg-red-500'];
                      const bgColor = colors[index] || 'bg-gray-600';
                      
                      return (
                        <div key={key} className="text-center">
                          <div className={`${bgColor} text-white px-2 py-1 rounded-full mb-1`}>
                            {featureNum} Feature{featureNum !== '1' ? 's' : ''}
                          </div>
                          <div className="text-gray-300">{count} items</div>
                        </div>
                      );
                    })}
                </div>
                {searchInfo.total_available && (
                  <p className="text-purple-300 text-xs mt-2 text-center">
                    Total Available: {searchInfo.total_available} items
                  </p>
                )}
              </div>
            )}
            
            {/* Extracted Attributes */}
            {Object.keys(searchInfo.extracted_attributes).length > 0 && (
              <div className="mt-3">
                <p className="text-gray-400 text-xs mb-1">Extracted attributes:</p>
                <div className="flex flex-wrap gap-1">
                  {Object.entries(searchInfo.extracted_attributes).map(([key, value]) => (
                    <span key={key} className="px-2 py-1 bg-blue-600 text-white text-xs rounded-full">
                      {key}: {value}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500 mx-auto"></div>
            <p className="text-gray-400 mt-4 text-lg">AI is analyzing your request and searching through fashion database...</p>
            <p className="text-gray-500 mt-2 text-sm">
              {searchMode === 'progressive' ? 'Finding quick results...' : 'Searching all combinations for best results'}
            </p>
          </div>
        )}

        {/* Results Section */}
        {((searchMode === 'continuous' ? allResults.length > 0 : results.length > 0) || expandedResults.length > 0) && !loading && (
          <div>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">
                {searchMode === 'continuous' ? 'Fashion Search Results (Priority Order)' : 'Search Results'}
                {searchInfo?.is_progressive && (
                  <span className="text-sm font-normal text-gray-400 ml-2">
                    (Quick results - {searchInfo.remaining_combinations} more combinations available)
                  </span>
                )}
              </h2>
              <div className="bg-gray-800 px-4 py-2 rounded-lg">
                <span className="text-gray-400 text-sm">
                  {searchMode === 'continuous' 
                    ? `${allResults.length} items found${searchInfo?.total_available ? ` of ${searchInfo.total_available}` : ''}`
                    : `${results.length + expandedResults.length} items found`
                  }
                </span>
              </div>
            </div>

            {/* Continuous Mode Results */}
            {searchMode === 'continuous' && allResults.length > 0 && (
              <div className="mb-8">
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {allResults.map((item) => (
                    <div
                      key={`continuous-${item.id}`}
                      className="bg-gray-800/50 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-700 hover:border-gray-600 transition-all duration-300 group"
                    >
                      <div className="relative">
                        <img
                          src={item.image}
                          alt={item.name}
                          className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.src = 'https://via.placeholder.com/256x256?text=Image+Not+Found';
                          }}
                        />
                        <div className="absolute top-2 right-2">
                          <button className="p-2 bg-gray-800/80 hover:bg-gray-700 rounded-full transition-colors">
                            <Heart className="w-4 h-4 text-white hover:text-red-500" />
                          </button>
                        </div>
                        {item.combination_type && (
                          <div className={`absolute top-2 left-2 text-white text-xs px-2 py-1 rounded-full ${
                            item.combination_type === 'full' ? 'bg-purple-600' :
                            item.combination_type === 'triple' ? 'bg-pink-600' :
                            item.combination_type === 'pair' ? 'bg-blue-600' :
                            'bg-green-500'
                          }`}>
                            {Object.keys(item.matching_filters || {}).length || 1} features
                          </div>
                        )}
                      </div>
                      
                      <div className="p-4">
                        <h3 className="text-white font-semibold text-lg mb-1">{item.name}</h3>
                        <p className="text-gray-400 text-sm mb-2">{item.category}</p>
                        
                        {item.matching_filters && (
                          <div className="mb-3">
                            <p className="text-gray-500 text-xs mb-1">Matched criteria:</p>
                            <div className="flex flex-wrap gap-1">
                              {Object.entries(item.matching_filters).map(([key, value]) => (
                                <span key={key} className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">
                                  {String(value)}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        <p className="text-gray-500 text-xs mb-3">{item.reason}</p>
                        
                        <button className="w-full py-2 bg-gradient-to-r from-purple-500 to-pink-600 text-white font-semibold rounded-lg hover:from-purple-600 hover:to-pink-700 transition-all duration-300 flex items-center justify-center">
                          <ShoppingBag className="w-4 h-4 mr-1" />
                          View Details
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Show More Button for Continuous Mode */}
                {hasMoreResults && (
                  <div className="flex justify-center mt-8">
                    <button
                      onClick={handleShowMore}
                      disabled={isExpanding}
                      className={`px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-600 text-white font-semibold rounded-lg hover:from-purple-600 hover:to-pink-700 transition-all duration-300 flex items-center ${
                        isExpanding ? 'opacity-50 cursor-not-allowed' : ''
                      }`}
                    >
                      {isExpanding ? (
                        <>
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                          Loading More...
                        </>
                      ) : (
                        <>
                          <Search className="w-5 h-5 mr-2" />
                          Show More
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Progressive/Complete Mode Results */}
            {searchMode !== 'continuous' && results.length > 0 && (
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  Best Matches
                </h3>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {results.map((item) => (
                    <div
                      key={item.id}
                      className="bg-gray-800/50 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-700 hover:border-gray-600 transition-all duration-300 group"
                    >
                      <div className="relative">
                        <img
                          src={item.image}
                          alt={item.name}
                          className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.src = 'https://via.placeholder.com/256x256?text=Image+Not+Found';
                          }}
                        />
                        <div className="absolute top-2 right-2">
                          <button className="p-2 bg-gray-800/80 hover:bg-gray-700 rounded-full transition-colors">
                            <Heart className="w-4 h-4 text-white hover:text-red-500" />
                          </button>
                        </div>
                        {item.combination_type && (
                          <div className={`absolute top-2 left-2 text-white text-xs px-2 py-1 rounded-full ${
                            item.combination_type === 'full' ? 'bg-purple-600' :
                            item.combination_type === 'triple' ? 'bg-pink-600' :
                            item.combination_type === 'pair' ? 'bg-blue-600' :
                            'bg-orange-500'
                          }`}>
                            {item.combination_type} match
                          </div>
                        )}
                      </div>
                      
                      <div className="p-4">
                        <h3 className="text-white font-semibold text-lg mb-1">{item.name}</h3>
                        <p className="text-gray-400 text-sm mb-2">{item.category}</p>
                        
                        {item.matching_filters && (
                          <div className="mb-3">
                            <p className="text-gray-500 text-xs mb-1">Matched criteria:</p>
                            <div className="flex flex-wrap gap-1">
                              {Object.entries(item.matching_filters).map(([key, value]) => (
                                <span key={key} className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded">
                                  {String(value)}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        <p className="text-gray-500 text-xs mb-3">{item.reason}</p>
                        
                        <button className="w-full py-2 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 flex items-center justify-center">
                          <ShoppingBag className="w-4 h-4 mr-1" />
                          View Details
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Search More Button (for non-continuous modes) */}
            {searchMode !== 'continuous' && canExpandSearch && !isExpanding && (
              <div className="flex justify-center mb-8">
                <button
                  onClick={handleSearchMore}
                  className="px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-300 flex items-center"
                >
                  <Search className="w-5 h-5 mr-2" />
                  Search More (Broader Combinations)
                </button>
              </div>
            )}

            {/* Expanding Loader */}
            {searchMode !== 'continuous' && isExpanding && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                <p className="text-gray-400 mt-4">Finding more matches with broader search combinations...</p>
              </div>
            )}

            {/* Expanded Results (for non-continuous modes) */}
            {searchMode !== 'continuous' && expandedResults.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                  <Search className="w-5 h-5 text-blue-500 mr-2" />
                  More Matches (Broader Combinations)
                </h3>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {expandedResults.map((item) => (
                    <div
                      key={`expanded-${item.id}`}
                      className="bg-gray-800/30 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-600 hover:border-gray-500 transition-all duration-300 group"
                    >
                      <div className="relative">
                        <img
                          src={item.image}
                          alt={item.name}
                          className="w-full h-48 object-cover group-hover:scale-105 transition-transform duration-300"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.src = 'https://via.placeholder.com/256x256?text=Image+Not+Found';
                          }}
                        />
                        <div className="absolute top-2 right-2">
                          <button className="p-2 bg-gray-800/80 hover:bg-gray-700 rounded-full transition-colors">
                            <Heart className="w-4 h-4 text-white hover:text-red-500" />
                          </button>
                        </div>
                        {item.combination_type && (
                          <div className={`absolute top-2 left-2 text-white text-xs px-2 py-1 rounded-full ${
                            item.combination_type === 'pair' ? 'bg-blue-500' :
                            item.combination_type === 'individual' ? 'bg-green-500' :
                            'bg-gray-500'
                          }`}>
                            {item.combination_type} match
                          </div>
                        )}
                      </div>
                      
                      <div className="p-4">
                        <h3 className="text-white font-semibold text-lg mb-1">{item.name}</h3>
                        <p className="text-gray-400 text-sm mb-2">{item.category}</p>
                        
                        {item.matching_filters && (
                          <div className="mb-3">
                            <p className="text-gray-500 text-xs mb-1">Matched criteria:</p>
                            <div className="flex flex-wrap gap-1">
                              {Object.entries(item.matching_filters).map(([key, value]) => (
                                <span key={key} className="text-xs bg-gray-600 text-gray-300 px-2 py-1 rounded">
                                  {String(value)}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        <p className="text-gray-500 text-xs mb-3">{item.reason}</p>
                        
                        <button className="w-full py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-300 flex items-center justify-center">
                          <ShoppingBag className="w-4 h-4 mr-1" />
                          View Details
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {((searchMode === 'continuous' ? allResults.length === 0 : results.length === 0) && expandedResults.length === 0) && !loading && query && (
          <div className="text-center py-12">
            <Search className="w-20 h-20 text-gray-500 mx-auto mb-6" />
            <h3 className="text-xl font-semibold text-white mb-2">No Results Found</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Try adjusting your search terms or using different keywords. Our AI searches through multiple combinations to find the best matches.
            </p>
          </div>
        )}

        {/* Initial State */}
        {((searchMode === 'continuous' ? allResults.length === 0 : results.length === 0) && expandedResults.length === 0) && !loading && !query && (
          <div className="text-center py-12"> 
            <h3 className="text-xl font-semibold text-white mb-2">Ready to Search</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Describe any clothing item you're looking for. Our AI will use intelligent search combinations to find the perfect matches for you.
            </p>
          </div>
        )}

        {/* Error State */}
        {searchInfo?.error && (
          <div className="bg-red-900/50 border border-red-700 rounded-xl p-4 text-red-300 mb-8">
            Error: {searchInfo.error}
          </div>
        )}
      </div>
    </div>
  );
}

export default AISearch;
