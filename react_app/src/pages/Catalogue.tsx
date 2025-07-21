import { useState } from 'react';
import { Search, Heart, ShoppingBag, Grid3X3, List } from 'lucide-react';

interface FashionItem {
  id: string;
  name: string;
  category: string;
  image: string;
  dominant_color: string;
  sleeve_type: string;
  pattern_type: string;
  description: string;
}

const Catalogue = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [likedItems, setLikedItems] = useState<Set<string>>(new Set());

  const categories = ['All', 'Long Sleeve', 'Short Sleeve', 'Sleeveless'];

  // Hardcoded fashion items from your local clothes directory
  const clothingItems: FashionItem[] = [
    {
      id: "00006_00",
      name: "00006_00",
      category: "Long Sleeve",
      image: "http://localhost:8001/static/clothes/00006_00.jpg",
      dominant_color: "Black",
      sleeve_type: "Long Sleeve",
      pattern_type: "Solid",
      description: "Beautiful Black Dress for casual wear"
    },
    {
      id: "00008_00",
      name: "00008_00",
      category: "Long Sleeve",
      image: "http://localhost:8001/static/clothes/00008_00.jpg",
      dominant_color: "Red",
      sleeve_type: "Long Sleeve",
      pattern_type: "Solid",
      description: "Stretchable and comfortable red dress"
    },
    {
      id: "00013_00",
      name: "00013_00",
      category: "Short Sleeve",
      image: "http://localhost:8001/static/clothes/00013_00.jpg",
      dominant_color: "Pink",
      sleeve_type: "Short Sleeve",
      pattern_type: "Solid",
      description: "Light and comfortable summer top"
    },
    {
      id: "00017_00",
      name: "00017_00",
      category: "Short Sleeve",
      image: "http://localhost:8001/static/clothes/00017_00.jpg",
      dominant_color: "Red",
      sleeve_type: "Short Sleeve",
      pattern_type: "Solid",
      description: "Elegant red dress for special occasions"
    },
    {
      id: "00034_00",
      name: "00034_00",
      category: "Sleeveless",
      image: "http://localhost:8001/static/clothes/00034_00.jpg",
      dominant_color: "Pink",
      sleeve_type: "Sleeveless",
      pattern_type: "Solid",
      description: "Classic denim jacket for casual outings"
    },
    {
      id: "00035_00",
      name: "00035_00",
      category: "Long Sleeve",
      image: "http://localhost:8001/static/clothes/00035_00.jpg",
      dominant_color: "Black",
      sleeve_type: "Long Sleeve",
      pattern_type: "Graphic Pattern",
      description: "Stylish long sleeve top for everyday wear"
    },
    {
      id: "00055_00",
      name: "00055_00",
      category: "Short Sleeve",
      image: "http://localhost:8001/static/clothes/00055_00.jpg",
      dominant_color: "Purple",
      sleeve_type: "Short Sleeve",
      pattern_type: "Graphic Pattern",
      description: "Elegant purple dress for evening events"
    },
    {
      id: "00057_00",
      name: "00057_00",
      category: "Short Sleeve",
      image: "http://localhost:8001/static/clothes/00057_00.jpg",
      dominant_color: "Black",
      sleeve_type: "Short Sleeve",
      pattern_type: "Graphic Pattern",
      description: "Comfortable top for daily wear"
    }
  ];

  const filteredItems = clothingItems.filter(item => {
    const matchesCategory = selectedCategory === 'All' || item.category === selectedCategory;
    const matchesSearch = item.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         item.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         item.description.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const toggleLike = (itemId: string) => {
    const newLiked = new Set(likedItems);
    if (newLiked.has(itemId)) {
      newLiked.delete(itemId);
    } else {
      newLiked.add(itemId);
    }
    setLikedItems(newLiked);
  };

  const getColorStyle = (color: string) => {
    const colorMap: { [key: string]: string } = {
      'White': '#ffffff',
      'Black': '#000000',
      'Red': '#ef4444',
      'Blue': '#3b82f6',
      'Green': '#10b981',
      'Pink': '#ec4899',
      'Yellow': '#eab308',
      'Navy': '#1e3a8a',
      'Gray': '#6b7280',
      'Brown': '#a16207',
      'Tan': '#d2b48c',
      'Light Blue': '#7dd3fc',
      'Burgundy': '#7c2d12',
      'Cream': '#fef3c7',
      'Nude': '#d4a574',
      'Floral': 'linear-gradient(45deg, #ec4899, #10b981)',
      'Purple': '#8b5cf6',
      'Orange': '#f97316',
      'Violet': '#7c3aed'
    };
    return { backgroundColor: colorMap[color] || '#6b7280' };
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl lg:text-5xl font-bold text-white mb-4">
            Fashion Gallery
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Discover our curated collection and try them on virtually. Find your perfect style from our diverse range of fashion pieces.
          </p>
        </div>

        {/* Search and Filters */}
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 mb-8">
          <div className="flex flex-col lg:flex-row gap-4 items-center justify-between">
            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search fashion items..."
                className="w-full pl-10 pr-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
              />
            </div>

            {/* Category Filter */}
            <div className="flex items-center space-x-2">
              {categories.map((category) => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    selectedCategory === category
                      ? 'bg-gradient-to-r from-orange-500 to-red-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded-lg transition-colors ${
                  viewMode === 'grid' ? 'bg-orange-500 text-white' : 'bg-gray-700 text-gray-400'
                }`}
              >
                <Grid3X3 className="w-5 h-5" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded-lg transition-colors ${
                  viewMode === 'list' ? 'bg-orange-500 text-white' : 'bg-gray-700 text-gray-400'
                }`}
              >
                <List className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Results Count */}
        <div className="flex items-center justify-between mb-6">
          <p className="text-gray-400">
            Showing {filteredItems.length} of {clothingItems.length} items
          </p>
        </div>

        {/* Clothing Grid */}
        {viewMode === 'grid' ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="bg-gray-800/50 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-700 hover:border-gray-600 transition-all duration-300 group"
              >
                <div className="relative aspect-[3/4] overflow-hidden">
                  <img
                    src={item.image}
                    alt={item.name}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = 'https://via.placeholder.com/400x600/374151/9CA3AF?text=No+Image';
                    }}
                  />
                  <button
                    onClick={() => toggleLike(item.id)}
                    className="absolute top-3 right-3 p-2 bg-gray-900/70 rounded-full hover:bg-gray-800 transition-colors"
                  >
                    <Heart
                      className={`w-4 h-4 ${
                        likedItems.has(item.id) ? 'text-red-500 fill-red-500' : 'text-white'
                      }`}
                    />
                  </button>
                </div>
                
                <div className="p-4">
                  <h3 className="text-white font-semibold mb-1">{item.name}</h3>
                  <p className="text-gray-400 text-sm mb-2">{item.category}</p>
                  
                  <div className="flex items-center space-x-2 mb-3">
                    <div className="flex items-center space-x-1">
                      <div
                        className="w-4 h-4 rounded-full border border-gray-600"
                        style={getColorStyle(item.dominant_color)}
                        title={`Color: ${item.dominant_color}`}
                      />
                      <span className="text-xs text-gray-400">{item.dominant_color}</span>
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-500 mb-3 space-y-1">
                    {item.sleeve_type && item.sleeve_type !== 'N/A' && <div>Sleeve: {item.sleeve_type}</div>}
                    {item.pattern_type && <div>Pattern: {item.pattern_type}</div>}
                  </div>
                  
                  <button className="w-full py-2 bg-gradient-to-r from-orange-500 to-red-600 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300 flex items-center justify-center">
                    <ShoppingBag className="w-4 h-4 mr-1" />
                    Try On
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {filteredItems.map((item) => (
              <div
                key={item.id}
                className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 hover:border-gray-600 transition-all duration-300 p-4"
              >
                <div className="flex items-center space-x-4">
                  <div className="w-24 h-32 overflow-hidden rounded-lg">
                    <img
                      src={item.image}
                      alt={item.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src = 'https://via.placeholder.com/200x250/374151/9CA3AF?text=No+Image';
                      }}
                    />
                  </div>
                  
                  <div className="flex-1">
                    <h3 className="text-white font-semibold text-lg mb-1">{item.name}</h3>
                    <p className="text-gray-400 text-sm mb-2">{item.category}</p>
                    <p className="text-gray-300 text-sm mb-3">{item.description}</p>
                    
                    <div className="flex items-center space-x-4 mb-3 text-xs text-gray-400">
                      <div className="flex items-center space-x-1">
                        <div
                          className="w-3 h-3 rounded-full border border-gray-600"
                          style={getColorStyle(item.dominant_color)}
                        />
                        <span>{item.dominant_color}</span>
                      </div>
                      {item.sleeve_type && item.sleeve_type !== 'N/A' && <span>Sleeve: {item.sleeve_type}</span>}
                      {item.pattern_type && <span>Pattern: {item.pattern_type}</span>}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => toggleLike(item.id)}
                      className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                    >
                      <Heart
                        className={`w-4 h-4 ${
                          likedItems.has(item.id) ? 'text-red-500 fill-red-500' : 'text-white'
                        }`}
                      />
                    </button>
                    <button className="flex items-center px-4 py-2 bg-gradient-to-r from-orange-500 to-red-600 text-white text-sm rounded-lg hover:from-orange-600 hover:to-red-700 transition-all duration-300">
                      <ShoppingBag className="w-4 h-4 mr-1" />
                      Try On
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Empty State */}
        {filteredItems.length === 0 && (
          <div className="text-center py-12">
            <Search className="w-16 h-16 text-gray-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-white mb-2">No items found</h3>
            <p className="text-gray-400">Try adjusting your search or filter criteria</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Catalogue;