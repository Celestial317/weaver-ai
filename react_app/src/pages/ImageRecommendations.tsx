import { useState } from 'react';
import { Upload, User, Heart, ShoppingBag, AlertCircle, Palette } from 'lucide-react';
import FileUpload from '../components/FileUpload';
import LoadingSpinner from '../components/LoadingSpinner';

const ImageRecommendations = () => {
  const [userImage, setUserImage] = useState<File | null>(null);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [themeMessage, setThemeMessage] = useState<string | null>(null);
  const [selectedLimit, setSelectedLimit] = useState<number>(10);

  const analyzeImage = async () => {
    if (!userImage) {
      setError('Please upload your photo first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setRecommendations([]); // Clear previous recommendations
    setThemeMessage(null); // Clear previous theme

    try {
      const formData = new FormData();
      formData.append('file', userImage);
      formData.append('limit', selectedLimit.toString());

      const response = await fetch(`http://localhost:8001/recommend?limit=${selectedLimit}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.error) {
        setError(result.error);
        setRecommendations([]);
        setThemeMessage(null);
      } else {
        setRecommendations(result.recommendations || []);
        setThemeMessage(result.theme);
        setError(null);
      }
      
    } catch (err) {
      console.error('Error calling API:', err);
      setError('Please provide a clear, well-lit image of your face for accurate style recommendations. Ensure good lighting and avoid heavy filters.');
      setRecommendations([]);
      setThemeMessage(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl lg:text-5xl font-bold text-white mb-4">
            Thematic Recommendations
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Upload your facial photo and get personalized fashion recommendations based on AI analysis of your style, vibe and facial structure.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 sticky top-24">
              <div className="flex items-center mb-4">
                <User className="w-6 h-6 text-orange-500 mr-3" />
                <h3 className="text-xl font-semibold text-white">Upload Your Photo</h3>
              </div>
              
              <FileUpload
                onFileSelect={setUserImage}
                accept="image/*"
                placeholder="Upload your photo for personalized recommendations"
                icon={Upload}
              />
              
              {userImage && (
                <div className="mt-4 p-3 bg-gray-700/50 rounded-lg">
                  <p className="text-sm text-gray-300">✓ Photo uploaded: {userImage.name}</p>
                </div>
              )}

              {/* Number of Recommendations Dropdown */}
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Number of Recommendations
                </label>
                <select
                  value={selectedLimit}
                  onChange={(e) => setSelectedLimit(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                >
                  <option value={5}>5 recommendations</option>
                  <option value={10}>10 recommendations</option>
                  <option value={15}>15 recommendations</option>
                  <option value={20}>20 recommendations</option>
                </select>
              </div>

              <button
                onClick={analyzeImage}
                disabled={!userImage || isAnalyzing}
                className="w-full mt-6 py-4 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {isAnalyzing ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span className="ml-2">Analyzing Your Style...</span>
                  </>
                ) : (
                  'Get My Recommendations'
                )}
              </button>

              {error && (
                <div className="flex items-center p-4 mt-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-3" />
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}

              {/* Analysis Tips */}
              <div className="mt-6 p-4 bg-gray-700/30 rounded-lg">
                <h4 className="text-sm font-semibold text-white mb-2">Tips for best results:</h4>
                <ul className="text-xs text-gray-400 space-y-1">
                  <li>• Use a clear, well-lit photo</li>
                  <li>• Face the camera directly</li>
                  <li>• Show your full outfit if possible</li>
                  <li>• Avoid heavy filters or editing</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {isAnalyzing && (
              <div className="text-center py-20">
                <LoadingSpinner size="lg" />
                <p className="text-gray-400 mt-6 text-lg">Analyzing your photo and finding perfect matches...</p>
                <p className="text-gray-500 mt-2 text-sm">This may take a few moments</p>
              </div>
            )}

            {recommendations.length > 0 && !isAnalyzing && (
              <div>
                {/* Theme Display */}
                {themeMessage && (
                  <div className="mb-6 p-4 bg-gradient-to-r from-orange-500/10 to-red-600/10 border border-orange-500/20 rounded-xl">
                    <div className="flex items-center mb-2">
                      <Palette className="w-5 h-5 text-orange-500 mr-2" />
                      <h3 className="text-lg font-semibold text-white">Your Color Palette Today</h3>
                    </div>
                    <p className="text-gray-300">{themeMessage}</p>
                  </div>
                )}

                <div className="flex items-center justify-between mb-8">
                  <h2 className="text-2xl font-bold text-white">Personalized Recommendations</h2>
                  <div className="bg-gray-800 px-4 py-2 rounded-lg">
                    <span className="text-gray-400 text-sm">{recommendations.length} perfect matches</span>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  {recommendations.map((item) => (
                    <div
                      key={item.id}
                      className="bg-gray-800/50 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-700 hover:border-gray-600 transition-all duration-300 group"
                    >
                      <div className="flex">
                        <div className="relative w-32 h-40 overflow-hidden">
                          <img
                            src={item.image}
                            alt={item.name}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                          />
                          <div className="absolute top-2 left-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full">
                            {item.matchScore}% match
                          </div>
                        </div>
                        
                        <div className="flex-1 p-4">
                          <div className="flex items-start justify-between mb-2">
                            <div>
                              <h3 className="text-white font-semibold text-lg">{item.name}</h3>
                              <p className="text-gray-400 text-sm">{item.category}</p>
                            </div>
                            <button className="p-2 hover:bg-gray-700 rounded-full transition-colors">
                              <Heart className="w-4 h-4 text-gray-400 hover:text-red-500" />
                            </button>
                          </div>
                          
                          <p className="text-gray-300 text-sm mb-3">{item.reason}</p>
                          
                          <button className="w-full py-2 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 flex items-center justify-center">
                            <ShoppingBag className="w-4 h-4 mr-1" />
                            Try On
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Error State - Face Detection Failed */}
            {error && !isAnalyzing && (
              <div className="text-center py-20">
                <AlertCircle className="w-20 h-20 text-red-500 mx-auto mb-6" />
                <h3 className="text-xl font-semibold text-white mb-4">Image Analysis Failed</h3>
                <div className="max-w-md mx-auto p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <p className="text-red-400">{error}</p>
                </div>
                <div className="mt-6">
                  <p className="text-gray-400 text-sm mb-4">Tips for better results:</p>
                  <ul className="text-gray-500 text-sm space-y-1 max-w-sm mx-auto text-left">
                    <li>• Ensure your face is clearly visible</li>
                    <li>• Use good lighting (natural light works best)</li>
                    <li>• Face the camera directly</li>
                    <li>• Avoid sunglasses or face coverings</li>
                    <li>• Remove heavy filters or effects</li>
                  </ul>
                </div>
              </div>
            )}

            {/* Empty State */}
            {recommendations.length === 0 && !isAnalyzing && !error && (
              <div className="text-center py-20">
                <User className="w-20 h-20 text-gray-500 mx-auto mb-6" />
                <h3 className="text-xl font-semibold text-white mb-2">Ready for AI Analysis</h3>
                <p className="text-gray-400 max-w-md mx-auto">
                  Upload your photo to get personalized style recommendations powered by advanced AI technology.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageRecommendations;