import { Link } from 'react-router-dom';
import { ArrowRight, Sparkles, Camera, Palette, Search, Grid3X3, Eye } from 'lucide-react';

const HomePage = () => {
  const features = [
    {
      icon: Camera,
      title: 'Virtual Try-On',
      description: 'Experience fashion like never before with our AI-powered virtual try-on technology',
      link: '/virtual-tryon',
      gradient: 'from-blue-500 to-cyan-600'
    },
    {
      icon: Sparkles,
      title: 'AI Search',
      description: 'Describe your style and let AI find the perfect outfit combinations for you',
      link: '/ai-search',
      gradient: 'from-purple-500 to-pink-600'
    },
    {
      icon: Search,
      title: 'Smart Recommendations',
      description: 'Upload your photo and get personalized style recommendations',
      link: '/image-recommendations',
      gradient: 'from-green-500 to-emerald-600'
    },
    {
      icon: Palette,
      title: 'Image Stylizer',
      description: 'Transform your images with AI-powered style transfer technology',
      link: '/ai-stylizer',
      gradient: 'from-orange-500 to-red-600'
    },
    {
      icon: Eye,
      title: 'Visual Designer',
      description: 'Find similar fashion items using CLIP-powered visual similarity search',
      link: '/visual-designer',
      gradient: 'from-indigo-500 to-purple-600'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Hero Section */}
      <section className="relative px-4 py-20 lg:py-32">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-5xl lg:text-7xl font-bold text-white mb-6 leading-tight">
            Experience Fashion
            <br />
            with <span className="bg-gradient-to-r from-orange-500 to-red-600 bg-clip-text text-transparent">Elegance</span>
          </h1>
          
          <p className="text-xl text-gray-400 mb-12 max-w-3xl mx-auto leading-relaxed">
            Try on clothes virtually, get AI-powered style recommendations, 
            and discover your perfect look with sophisticated technology.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/catalogue"
              className="px-8 py-4 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 transform hover:scale-105 flex items-center justify-center"
            >
              Explore Collection
              <ArrowRight className="ml-2 w-5 h-5" />
            </Link>
            <Link
              to="/virtual-tryon"
              className="px-8 py-4 bg-gray-800 text-white font-semibold rounded-lg hover:bg-gray-700 transition-all duration-300 border border-gray-700 flex items-center justify-center"
            >
              Start Try-On
            </Link>
          </div>
        </div>
        
        {/* Background decoration */}
        <div className="absolute top-20 left-10 w-20 h-20 bg-orange-500/20 rounded-full blur-xl"></div>
        <div className="absolute bottom-20 right-10 w-32 h-32 bg-red-500/20 rounded-full blur-xl"></div>
      </section>

      {/* Features Section */}
      <section className="px-4 py-20 bg-gray-900/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">Powerful AI Features</h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Discover the future of fashion with our comprehensive suite of AI-powered tools
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-8">
            {features.map((feature, index) => (
              <Link
                key={index}
                to={feature.link}
                className="group bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition-all duration-300 hover:transform hover:scale-105"
              >
                <div className={`w-12 h-12 bg-gradient-to-r ${feature.gradient} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400 text-sm leading-relaxed">{feature.description}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Access Section */}
      <section className="px-4 py-20">
        <div className="max-w-4xl mx-auto">
          <div className="bg-gradient-to-r from-orange-500/10 to-red-600/10 rounded-2xl p-8 border border-orange-500/20">
            <div className="text-center">
              <Grid3X3 className="w-16 h-16 text-orange-500 mx-auto mb-6" />
              <h2 className="text-3xl font-bold text-white mb-4">Ready to Transform Your Style?</h2>
              <p className="text-xl text-gray-300 mb-8">
                Join thousands of fashion enthusiasts who are already using Weaver AI to discover their perfect look
              </p>
              <Link
                to="image-recommendations"
                className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 transform hover:scale-105"
              >
                Start Your Style Journey
                <Sparkles className="ml-2 w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;