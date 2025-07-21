import React, { useState } from 'react';
import { Upload, Camera, Download, Loader, AlertCircle } from 'lucide-react';
import FileUpload from '../components/FileUpload';
import LoadingSpinner from '../components/LoadingSpinner';

const VirtualTryOn = () => {
  const [userPhoto, setUserPhoto] = useState<File | null>(null);
  const [garmentPhoto, setGarmentPhoto] = useState<File | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleTryOn = async () => {
    if (!userPhoto || !garmentPhoto) {
      setError('Please upload both your photo and garment image');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Simulate API call to FastAPI backend
      const formData = new FormData();
      formData.append('user_photo', userPhoto);
      formData.append('garment_photo', garmentPhoto);

      // Simulated delay for demonstration
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Mock result - in real implementation, this would be the API response
      setResultImage('https://images.pexels.com/photos/1536619/pexels-photo-1536619.jpeg?auto=compress&cs=tinysrgb&w=800');
      
    } catch (err) {
      setError('Failed to process images. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadResult = () => {
    if (resultImage) {
      const link = document.createElement('a');
      link.href = resultImage;
      link.download = 'virtual-tryon-result.jpg';
      link.click();
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl lg:text-5xl font-bold text-white mb-4">
            Virtual Try-On Studio
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Experience fashion like never before. Upload your photo and try on any garment virtually.
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <div className="grid grid-cols-1 gap-6">
            {/* User Photo Upload */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <div className="flex items-center mb-4">
                <Camera className="w-6 h-6 text-orange-500 mr-3" />
                <h3 className="text-xl font-semibold text-white">Upload Your Photo</h3>
              </div>
              <FileUpload
                onFileSelect={setUserPhoto}
                accept="image/*"
                placeholder="Upload your photo (PNG, JPG up to 10MB)"
                icon={Upload}
              />
              {userPhoto && (
                <div className="mt-4 p-3 bg-gray-700/50 rounded-lg">
                  <p className="text-sm text-gray-300">✓ Photo uploaded: {userPhoto.name}</p>
                </div>
              )}
            </div>

            {/* Garment Photo Upload */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
              <div className="flex items-center mb-4">
                <Upload className="w-6 h-6 text-orange-500 mr-3" />
                <h3 className="text-xl font-semibold text-white">Upload Garment Photo</h3>
              </div>
              <FileUpload
                onFileSelect={setGarmentPhoto}
                accept="image/*"
                placeholder="Upload garment image (PNG, JPG up to 10MB)"
                icon={Upload}
              />
              {garmentPhoto && (
                <div className="mt-4 p-3 bg-gray-700/50 rounded-lg">
                  <p className="text-sm text-gray-300">✓ Garment uploaded: {garmentPhoto.name}</p>
                </div>
              )}
            </div>

            </div>
            
            {/* Generate Button */}
            <button
              className="w-full mt-6"
              onClick={handleTryOn}
              disabled={!userPhoto || !garmentPhoto || isProcessing}
              className="w-full py-4 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isProcessing ? (
                <>
                  <LoadingSpinner size="sm" />
                  <span className="ml-2">Generating Try-On...</span>
                </>
              ) : (
                'Generate Try-On'
              )}
            </button>

            {/* Error Message */}
            {error && (
              <div className="flex items-center p-4 mt-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-500 mr-3" />
                <p className="text-red-400">{error}</p>
              </div>
            )}
          </div>

          {/* Result Section */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">Live Preview</h3>
              {resultImage && (
                <button
                  onClick={downloadResult}
                  className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                >
                  <Download className="w-5 h-5 text-white" />
                </button>
              )}
            </div>

            <div className="aspect-[4/5] bg-gray-700/50 rounded-lg border-2 border-dashed border-gray-600 flex items-center justify-center">
              {isProcessing ? (
                <div className="text-center">
                  <LoadingSpinner size="lg" />
                  <p className="text-gray-400 mt-4">Processing your virtual try-on...</p>
                </div>
              ) : resultImage ? (
                <img
                  src={resultImage}
                  alt="Virtual try-on result"
                  className="w-full h-full object-cover rounded-lg"
                />
              ) : (
                <div className="text-center">
                  <Camera className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">Your virtual try-on will appear here</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VirtualTryOn;