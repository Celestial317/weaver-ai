import { useState } from 'react';
import { Upload, Palette, Sparkles, Heart, ShoppingBag, AlertCircle, Image as ImageIcon } from 'lucide-react';
import FileUpload from '../components/FileUpload';
import LoadingSpinner from '../components/LoadingSpinner';

interface VisualSearchResult {
  id: number;
  image_id: string;
  name: string;
  category: string;
  image: string;
  similarity_score: number;
  dominant_color: string;
  sleeve_type: string;
  pattern_type: string;
  description: string;
  metadata: Record<string, any>;
}

interface VisualSearchResponse {
  recommendations: VisualSearchResult[];
  total_found: number;
  query_info: {
    filename: string;
    size: [number, number];
    format?: string;
  };
  similarity_scores: number[];
  error: string | null;
}

const VisualDesigner = () => {
  const [queryImage, setQueryImage] = useState<File | null>(null);
  const [recommendations, setRecommendations] = useState<VisualSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchInfo, setSearchInfo] = useState<VisualSearchResponse | null>(null);
  const [selectedLimit, setSelectedLimit] = useState<number>(10);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const performVisualSearch = async () => {
    if (!queryImage) {
      setError('Please upload an image first');
      return;
    }

    setIsSearching(true);
    setError(null);
    setRecommendations([]);
    setSearchInfo(null);

    try {
      const formData = new FormData();
      formData.append('file', queryImage);
      formData.append('limit', selectedLimit.toString());

      const response = await fetch(`http://localhost:8004/visual-search?limit=${selectedLimit}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result: VisualSearchResponse = await response.json();
      
      if (result.error) {
        setError(result.error);
        setRecommendations([]);
        setSearchInfo(null);
      } else {
        setRecommendations(result.recommendations || []);
        setSearchInfo(result);
        setError(null);
      }
      
    } catch (err) {
      console.error('Error calling Visual Designer API:', err);
      setError('Failed to process image. Please ensure the image is clear and the Visual Designer service is running on port 8004.');
      setRecommendations([]);
      setSearchInfo(null);
    } finally {
      setIsSearching(false);
    }
  };

  const handleFileSelect = (file: File | null) => {
    setQueryImage(file);
    if (file) {
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  const formatSimilarityScore = (score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    return 'text-orange-400';
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center items-center mb-4">
            <h1 className="text-4xl lg:text-5xl font-bold text-white">
              Visual Designer
            </h1>
          </div>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Upload any fashion image and discover visually similar items using advanced AI. 
            Find styles that match your aesthetic preferences instantly.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 sticky top-24">
              <div className="flex items-center mb-4">
                <ImageIcon className="w-6 h-6 text-purple-500 mr-3" />
                <h3 className="text-xl font-semibold text-white">Upload Reference Image</h3>
              </div>
              
              <FileUpload
                onFileSelect={handleFileSelect}
                accept="image/*"
                placeholder="Upload a fashion image to find similar styles"
                icon={Upload}
              />
              
              {queryImage && (
                <div className="mt-4">
                  {/* Image Preview */}
                  {previewUrl && (
                    <div className="mb-3">
                      <img 
                        src={previewUrl} 
                        alt="Query preview" 
                        className="w-full h-32 object-cover rounded-lg border border-gray-600"
                      />
                    </div>
                  )}
                  <div className="p-3 bg-gray-700/50 rounded-lg">
                    <p className="text-sm text-gray-300">✓ Image uploaded: {queryImage.name}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Size: {Math.round(queryImage.size / 1024)}KB
                    </p>
                  </div>
                </div>
              )}

              {/* Number of Results Dropdown */}
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  <Sparkles className="w-4 h-4 inline mr-1" />
                  Number of Similar Items
                </label>
                <select
                  value={selectedLimit}
                  onChange={(e) => setSelectedLimit(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                >
                  <option value={5}>5 similar items</option>
                  <option value={10}>10 similar items</option>
                  <option value={15}>15 similar items</option>
                  <option value={20}>20 similar items</option>
                  <option value={30}>30 similar items</option>
                </select>
              </div>

              <button
                onClick={performVisualSearch}
                disabled={!queryImage || isSearching}
                className="w-full mt-6 py-4 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {isSearching ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span className="ml-2">Finding Similar Styles...</span>
                  </>
                ) : (
                  <>
                    Find Similar Items
                  </>
                )}
              </button>

              {error && (
                <div className="flex items-center p-4 mt-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-3" />
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}

              {/* Search Info */}
              {searchInfo && (
                <div className="mt-4 p-4 bg-purple-500/10 border border-purple-500/20 rounded-lg">
                  <h4 className="text-sm font-semibold text-purple-400 mb-2">
                    <Palette className="w-4 h-4 inline mr-1" />
                    Search Results
                  </h4>
                  <div className="text-xs text-gray-400 space-y-1">
                    <p>• Found {searchInfo.total_found} similar items</p>
                    <p>• Processed: {searchInfo.query_info.filename}</p>
                    <p>• Image size: {searchInfo.query_info.size[0]}×{searchInfo.query_info.size[1]}px</p>
                    {searchInfo.similarity_scores.length > 0 && (
                      <p>• Best match: {formatSimilarityScore(Math.max(...searchInfo.similarity_scores))}</p>
                    )}
                  </div>
                </div>
              )}

              {/* Tips */}
              <div className="mt-6 p-4 bg-gray-700/30 rounded-lg">
                <h4 className="text-sm font-semibold text-white mb-2">Tips for best results:</h4>
                <ul className="text-xs text-gray-400 space-y-1">
                  <li>• Use high-quality, clear images</li>
                  <li>• Focus on clothing items</li>
                  <li>• Avoid heavily edited photos</li>
                  <li>• Single item works better than outfits</li>
                  <li>• Good lighting improves accuracy</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {isSearching && (
              <div className="flex flex-col items-center justify-center py-16 bg-gray-800/30 rounded-xl border border-gray-700">
                <LoadingSpinner size="lg" />
                <p className="text-gray-300 mt-4 text-lg">Processing your image with AI...</p>
                <p className="text-gray-500 text-sm mt-2">Finding visually similar fashion items</p>
              </div>
            )}

            {!isSearching && recommendations.length === 0 && !error && (
              <div className="text-center py-16 bg-gray-800/30 rounded-xl border border-gray-700">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full flex items-center justify-center mx-auto mb-4">
                  <ImageIcon className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-gray-300 mb-2">Ready for Visual Search</h3>
                <p className="text-gray-500">Upload an image to discover visually similar fashion items</p>
              </div>
            )}

            {recommendations.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-white flex items-center">
                    <Sparkles className="w-6 h-6 text-purple-500 mr-2" />
                    Similar Fashion Items
                  </h2>
                  <span className="text-gray-400 text-sm">
                    {recommendations.length} items found
                  </span>
                </div>

                <div className="grid sm:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                  {recommendations.map((item, index) => (
                    <div
                      key={item.id}
                      className="group bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-4 hover:border-purple-500/50 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/10"
                    >
                      <div className="relative mb-4">
                        <img
                          src={item.image}
                          alt={item.name}
                          className="w-full h-48 object-cover rounded-lg"
                          onError={(e) => {
                            e.currentTarget.src = `https://via.placeholder.com/300x400/374151/9CA3AF?text=${item.image_id}`;
                          }}
                        />
                        <div className="absolute top-2 left-2">
                          <span className={`px-2 py-1 text-xs font-semibold rounded-full bg-gray-900/80 ${getScoreColor(item.similarity_score)}`}>
                            {formatSimilarityScore(item.similarity_score)} match
                          </span>
                        </div>
                        <div className="absolute top-2 right-2">
                          <span className="px-2 py-1 text-xs font-medium rounded-full bg-purple-500/80 text-white">
                            #{index + 1}
                          </span>
                        </div>
                      </div>
                      
                      <div className="space-y-2">
                        <h3 className="font-semibold text-white text-lg group-hover:text-purple-400 transition-colors">
                          {item.name}
                        </h3>
                        <p className="text-gray-400 text-sm">{item.description}</p>
                        
                        <div className="flex flex-wrap gap-2 text-xs">
                          <span className="px-2 py-1 bg-orange-500/20 text-orange-400 rounded-full">
                            {item.category}
                          </span>
                          {item.dominant_color && item.dominant_color !== 'Unknown' && (
                            <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full">
                              {item.dominant_color}
                            </span>
                          )}
                          {item.pattern_type && item.pattern_type !== 'Unknown' && (
                            <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded-full">
                              {item.pattern_type}
                            </span>
                          )}
                        </div>

                        <div className="flex items-center justify-between pt-3">
                          <span className="text-sm text-gray-500">ID: {item.image_id}</span>
                          <div className="flex space-x-2">
                            <button className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors">
                              <Heart className="w-4 h-4" />
                            </button>
                            <button className="p-2 text-gray-400 hover:text-orange-400 hover:bg-orange-500/10 rounded-lg transition-colors">
                              <ShoppingBag className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Summary Stats */}
                {searchInfo && searchInfo.similarity_scores.length > 0 && (
                  <div className="mt-8 p-6 bg-gray-800/30 rounded-xl border border-gray-700">
                    <h3 className="text-lg font-semibold text-white mb-4">Search Statistics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                      <div>
                        <p className="text-2xl font-bold text-purple-400">
                          {searchInfo.total_found}
                        </p>
                        <p className="text-sm text-gray-400">Items Found</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-green-400">
                          {formatSimilarityScore(Math.max(...searchInfo.similarity_scores))}
                        </p>
                        <p className="text-sm text-gray-400">Best Match</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-blue-400">
                          {formatSimilarityScore(searchInfo.similarity_scores.reduce((a, b) => a + b, 0) / searchInfo.similarity_scores.length)}
                        </p>
                        <p className="text-sm text-gray-400">Avg Similarity</p>
                      </div>
                      <div>
                        <p className="text-2xl font-bold text-orange-400">
                          {searchInfo.query_info.format || 'JPEG'}
                        </p>
                        <p className="text-sm text-gray-400">Image Format</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VisualDesigner;
