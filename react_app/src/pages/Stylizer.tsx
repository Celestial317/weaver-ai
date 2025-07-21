import { useState, useRef } from 'react';
import { Upload, Wand2, Download, RefreshCw, Image as ImageIcon } from 'lucide-react';

interface StyleTransferResult {
  success: boolean;
  message: string;
  result_url: string;
  result_path: string;
  base_image: string;
  style_image: string;
}

const Stylizer = () => {
  const [baseImage, setBaseImage] = useState<File | null>(null);
  const [styleImage, setStyleImage] = useState<File | null>(null);
  const [basePreview, setBasePreview] = useState<string | null>(null);
  const [stylePreview, setStylePreview] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<StyleTransferResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const baseFileRef = useRef<HTMLInputElement>(null);
  const styleFileRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (file: File, type: 'base' | 'style') => {
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const preview = e.target?.result as string;
      
      if (type === 'base') {
        setBaseImage(file);
        setBasePreview(preview);
      } else {
        setStyleImage(file);
        setStylePreview(preview);
      }
      
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handleStyleTransfer = async () => {
    if (!baseImage || !styleImage) {
      setError('Please select both base and style images');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('base_image', baseImage);
      formData.append('style_image', styleImage);
      formData.append('base_filename', baseImage.name);
      formData.append('style_filename', styleImage.name);

      const response = await fetch('http://localhost:8003/style-transfer', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: StyleTransferResult = await response.json();
      
      if (data.success) {
        setResult(data);
      } else {
        throw new Error(data.message || 'Style transfer failed');
      }

    } catch (err) {
      console.error('Style transfer error:', err);
      setError(err instanceof Error ? err.message : 'Failed to process style transfer');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (result?.result_url) {
      const link = document.createElement('a');
      link.href = `http://localhost:8003${result.result_url}`;
      link.download = `style_transfer_${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  const resetAll = () => {
    setBaseImage(null);
    setStyleImage(null);
    setBasePreview(null);
    setStylePreview(null);
    setResult(null);
    setError(null);
    if (baseFileRef.current) baseFileRef.current.value = '';
    if (styleFileRef.current) styleFileRef.current.value = '';
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <h1 className="text-4xl lg:text-5xl font-bold text-white">
              Fashion Stylizer
            </h1> 
          </div>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Transform your fashion pieces with AI-powered style transfer. Combine the structure of one garment with the pattern and style of another to create unique designs.
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-8 p-4 bg-red-900/50 border border-red-600 rounded-lg text-center">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* Upload Section */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Base Image Upload */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <ImageIcon className="w-5 h-5 mr-2 text-blue-400" />
              Base Image (Structure & Color)
            </h3>
            <p className="text-gray-400 mb-4 text-sm">
              Upload the garment that will provide the structure, shape, and base color for the final design.
            </p>
            
            <div 
              className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-gray-500 transition-colors cursor-pointer"
              onClick={() => baseFileRef.current?.click()}
            >
              {basePreview ? (
                <div className="space-y-4">
                  <img 
                    src={basePreview} 
                    alt="Base preview" 
                    className="max-h-64 mx-auto rounded-lg shadow-lg"
                  />
                  <p className="text-sm text-gray-400">{baseImage?.name}</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                  <div>
                    <p className="text-gray-300">Click to upload base image</p>
                    <p className="text-sm text-gray-500">PNG, JPG, JPEG up to 10MB</p>
                  </div>
                </div>
              )}
            </div>
            
            <input
              ref={baseFileRef}
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleImageUpload(file, 'base');
              }}
              className="hidden"
            />
          </div>

          {/* Style Image Upload */}
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Wand2 className="w-5 h-5 mr-2 text-purple-400" />
              Style Image (Pattern & Texture)
            </h3>
            <p className="text-gray-400 mb-4 text-sm">
              Upload the garment that will provide the pattern, texture, and style elements for the transfer.
            </p>
            
            <div 
              className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-gray-500 transition-colors cursor-pointer"
              onClick={() => styleFileRef.current?.click()}
            >
              {stylePreview ? (
                <div className="space-y-4">
                  <img 
                    src={stylePreview} 
                    alt="Style preview" 
                    className="max-h-64 mx-auto rounded-lg shadow-lg"
                  />
                  <p className="text-sm text-gray-400">{styleImage?.name}</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                  <div>
                    <p className="text-gray-300">Click to upload style image</p>
                    <p className="text-sm text-gray-500">PNG, JPG, JPEG up to 10MB</p>
                  </div>
                </div>
              )}
            </div>
            
            <input
              ref={styleFileRef}
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleImageUpload(file, 'style');
              }}
              className="hidden"
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
          <button
            onClick={handleStyleTransfer}
            disabled={!baseImage || !styleImage || isProcessing}
            className="flex items-center justify-center px-8 py-3 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Wand2 className="w-5 h-5 mr-2" />
                Create Style Transfer
              </>
            )}
          </button>

          <button
            onClick={resetAll}
            className="flex items-center justify-center px-8 py-3 bg-gray-700 text-white font-semibold rounded-lg hover:bg-gray-600 transition-colors"
          >
            <RefreshCw className="w-5 h-5 mr-2" />
            Reset All
          </button>
        </div>

        {/* Processing Status */}
        {isProcessing && (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 text-center mb-8">
            <div className="animate-pulse">
              <Wand2 className="w-12 h-12 text-orange-500 mx-auto mb-4 animate-bounce" />
              <h3 className="text-xl font-semibold text-white mb-2">AI Magic in Progress...</h3>
              <p className="text-gray-400">
                Our AI is analyzing your images and creating a beautiful style transfer. This may take a few moments.
              </p>
            </div>
          </div>
        )}

        {/* Result Section */}
        {result && (
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
            <h3 className="text-2xl font-semibold text-white mb-6 text-center flex items-center justify-center">
              Style Transfer Complete!
            </h3>
            
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Before - Base Image */}
              <div className="text-center">
                <h4 className="text-lg font-medium text-gray-300 mb-3">Base Image</h4>
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <img 
                    src={basePreview!} 
                    alt="Base" 
                    className="max-h-64 mx-auto rounded-lg shadow-lg"
                  />
                </div>
              </div>

              {/* Before - Style Image */}
              <div className="text-center">
                <h4 className="text-lg font-medium text-gray-300 mb-3">Style Image</h4>
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <img 
                    src={stylePreview!} 
                    alt="Style" 
                    className="max-h-64 mx-auto rounded-lg shadow-lg"
                  />
                </div>
              </div>

              {/* After - Result */}
              <div className="text-center">
                <h4 className="text-lg font-medium text-green-400 mb-3">Generated Result</h4>
                <div className="bg-gradient-to-br from-green-900/20 to-blue-900/20 rounded-lg p-4 border border-green-500/30">
                  <img 
                    src={`http://localhost:8003${result.result_url}`}
                    alt="Style transfer result" 
                    className="max-h-64 mx-auto rounded-lg shadow-lg border-2 border-green-400/50"
                  />
                </div>
              </div>
            </div>

            {/* Download Button */}
            <div className="text-center mt-6">
              <button
                onClick={handleDownload}
                className="inline-flex items-center px-6 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition-colors"
              >
                <Download className="w-5 h-5 mr-2" />
                Download Result
              </button>
            </div>
          </div>
        )}

        {/* Info Section */}
        <div className="mt-12 bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-semibold text-white mb-4">How It Works</h3>
          <div className="grid md:grid-cols-3 gap-6 text-sm text-gray-400">
            <div className="text-center">
              <div className="w-12 h-12 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-3">
                <ImageIcon className="w-6 h-6 text-blue-400" />
              </div>
              <h4 className="text-white font-medium mb-2">1. Upload Base Image</h4>
              <p>Choose the garment that will provide the structure and base color</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-purple-500/20 rounded-full flex items-center justify-center mx-auto mb-3">
                <Wand2 className="w-6 h-6 text-purple-400" />
              </div>
              <h4 className="text-white font-medium mb-2">2. Upload Style Image</h4>
              <p>Select the garment with the pattern or style you want to transfer</p>
            </div>
            <div className="text-center">
              <div className="w-12 h-12 bg-green-500/20 rounded-full flex items-center justify-center mx-auto mb-3">
              </div>
              <h4 className="text-white font-medium mb-2">3. AI Magic</h4>
              <p>Our AI combines both images to create a unique style-transferred result</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Stylizer;
